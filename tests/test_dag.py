from pathlib import Path
from textwrap import dedent

from ds_copilot.dag import (
    NotebookGraph,
    parse_notebook,
    parse_notebook_source,
    render_mermaid,
)


def _src(body: str) -> str:
    """Wrap cell-body source in a minimal marimo file scaffold."""
    header = (
        "import marimo\n"
        "\n"
        '__generated_with = "0.23.6"\n'
        'app = marimo.App(width="medium")\n'
        "\n"
    )
    footer = '\n\nif __name__ == "__main__":\n    app.run()\n'
    return header + body + footer


def test_empty_notebook() -> None:
    g = parse_notebook_source(_src(""))
    assert isinstance(g, NotebookGraph)
    assert g.cells == ()
    assert g.edges == ()


def test_single_cell_with_outputs() -> None:
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _():
            import polars as pl
            df = pl.DataFrame({"a": [1, 2]})
            return (df,)
        """)
    ))
    assert len(g.cells) == 1
    cell = g.cells[0]
    assert cell.inputs == ()
    assert cell.outputs == ("df",)
    # No edges in a single-cell graph
    assert g.edges == ()


def test_two_cell_dependency_chain() -> None:
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _():
            import polars as pl
            df = pl.DataFrame({"a": [1, 2]})
            return (df,)


        @app.cell
        def _(df):
            df.head()
            return
        """)
    ))
    assert len(g.cells) == 2
    assert g.edges == ((0, 1),)


def test_fan_out_and_fan_in() -> None:
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _():
            x = 1
            return (x,)


        @app.cell
        def _(x):
            y = x + 1
            return (y,)


        @app.cell
        def _(x):
            z = x * 2
            return (z,)


        @app.cell
        def _(y, z):
            total = y + z
            return (total,)
        """)
    ))
    edges = set(g.edges)
    assert (0, 1) in edges  # x -> y cell
    assert (0, 2) in edges  # x -> z cell
    assert (1, 3) in edges  # y -> total cell
    assert (2, 3) in edges  # z -> total cell
    assert len(edges) == 4


def test_markdown_cell_label_extracts_heading() -> None:
    g = parse_notebook_source(_src(
        dedent('''\
        @app.cell
        def _(mo):
            mo.md(r"""
            # Section heading
            with body text
            """)
            return


        @app.cell
        def _():
            import marimo as mo
            return (mo,)
        '''),
    ))
    md_cell = next(c for c in g.cells if c.kind == "md")
    assert "Section heading" in md_cell.label


def test_import_cell_classified() -> None:
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _():
            import marimo as mo
            import polars as pl
            return (mo, pl)
        """)
    ))
    assert g.cells[0].kind == "import"


def test_hide_code_decorator_picked_up() -> None:
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell(hide_code=True)
        def _():
            return


        @app.cell
        def _():
            return
        """)
    ))
    assert g.cells[0].hide_code is True
    assert g.cells[1].hide_code is False


def test_async_cell_is_parsed() -> None:
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _():
            planner = object()
            return (planner,)


        @app.cell
        async def _(planner):
            result = await planner.apply()
            return (result,)
        """)
    ))
    assert len(g.cells) == 2
    assert g.cells[1].inputs == ("planner",)
    assert g.cells[1].outputs == ("result",)
    assert (0, 1) in g.edges


def test_self_dependency_is_not_an_edge() -> None:
    """A cell that takes its own output as an arg shouldn't make a self-loop."""
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _(x):
            x = 1
            return (x,)
        """)
    ))
    # Even though `x` is both input and output, the cell can't depend on
    # itself, so no edge.
    assert g.edges == ()


def test_unused_outputs_are_fine() -> None:
    """Outputs not referenced downstream don't create edges."""
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _():
            unused = 99
            return (unused,)


        @app.cell
        def _():
            other = 7
            return (other,)
        """)
    ))
    assert g.edges == ()


def test_render_mermaid_produces_graph_td() -> None:
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _():
            df = 1
            return (df,)


        @app.cell
        def _(df):
            x = df + 1
            return
        """)
    ))
    diagram = render_mermaid(g)
    assert diagram.startswith("graph TD")
    assert "c0 --> c1" in diagram
    # Both nodes are declared
    assert "c0[" in diagram and "c1[" in diagram


def test_render_mermaid_escapes_label_brackets() -> None:
    g = parse_notebook_source(_src(
        dedent("""\
        @app.cell
        def _():
            data = [1, 2, 3]
            return (data,)
        """)
    ))
    diagram = render_mermaid(g)
    # Literal `[` in label would break mermaid; ensure it's been swapped
    for line in diagram.splitlines():
        if line.startswith("    c0["):
            # The cell-id wrapper still uses `[`, but the label body must not
            assert "[1, 2, 3]" not in line
            assert "(1, 2, 3)" in line
            break
    else:
        raise AssertionError("c0 node not found in diagram")


def test_parse_notebook_from_file(tmp_path: Path) -> None:
    """parse_notebook(path) reads the file off disk."""
    nb_path = tmp_path / "nb.py"
    nb_path.write_text(
        _src(
            dedent("""\
            @app.cell
            def _():
                a = 1
                return (a,)
            """)
        ),
        encoding="utf-8",
    )
    g = parse_notebook(nb_path)
    assert len(g.cells) == 1
    assert g.cells[0].outputs == ("a",)
