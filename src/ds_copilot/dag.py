"""Static DAG extraction for marimo notebooks.

Marimo `.py` files declare each cell as `@app.cell`-decorated function;
the function's positional args are the variable names the cell reads,
and its return tuple is the variable names it defines. Together those
make a directed acyclic graph: for every name `n` read by cell B and
defined by cell A, there is an edge A -> B.

This module parses a notebook file (no kernel required) and exposes
the graph + a mermaid rendering. Useful when notebooks grow past
~20 cells and you want to see what depends on what.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CellNode:
    """One cell from a marimo notebook."""

    index: int
    label: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    hide_code: bool = False
    kind: str = "code"  # "code", "md", "import"


@dataclass(frozen=True)
class NotebookGraph:
    cells: tuple[CellNode, ...]
    edges: tuple[tuple[int, int], ...]  # (upstream_index, downstream_index)


def parse_notebook(path: str | Path) -> NotebookGraph:
    """AST-parse a marimo .py file into a `NotebookGraph`."""
    src = Path(path).read_text(encoding="utf-8")
    return parse_notebook_source(src)


def parse_notebook_source(source: str) -> NotebookGraph:
    tree = ast.parse(source)

    cells: list[CellNode] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_app_cell_decorator(node.decorator_list):
            continue
        inputs = tuple(a.arg for a in node.args.args)
        outputs = _extract_return_names(node)
        label = _extract_label(node)
        hide_code = _has_hide_code(node.decorator_list)
        kind = _classify(node)
        cells.append(
            CellNode(
                index=len(cells),
                label=label,
                inputs=inputs,
                outputs=outputs,
                hide_code=hide_code,
                kind=kind,
            )
        )

    edges = tuple(_build_edges(cells))
    return NotebookGraph(cells=tuple(cells), edges=edges)


def render_mermaid(graph: NotebookGraph) -> str:
    """Render a `NotebookGraph` as a `graph TD` mermaid diagram string."""
    lines: list[str] = ["graph TD"]
    for c in graph.cells:
        label = _escape_mermaid_label(c.label)
        suffix = " *" if c.hide_code else ""
        lines.append(f'    c{c.index}["[{c.index}] {label}{suffix}"]')

    # Style nodes by kind. Mermaid `classDef` then `class c0,c1 className`.
    md_indices = [c.index for c in graph.cells if c.kind == "md"]
    import_indices = [c.index for c in graph.cells if c.kind == "import"]
    if md_indices:
        lines.append(
            "    classDef md fill:#f3f0ff,stroke:#7c3aed,color:#4c1d95;"
        )
        lines.append(f"    class {','.join(f'c{i}' for i in md_indices)} md;")
    if import_indices:
        lines.append(
            "    classDef imp fill:#f1f5f9,stroke:#475569,color:#0f172a;"
        )
        lines.append(
            f"    class {','.join(f'c{i}' for i in import_indices)} imp;"
        )

    for src, dst in graph.edges:
        lines.append(f"    c{src} --> c{dst}")
    return "\n".join(lines)


def dag_widget(notebook_path: str | Path) -> Any:
    """Render the cell DAG of a marimo notebook.

    Returns a Marimo display element (vstack of header + mermaid diagram).
    Pass an absolute or repo-relative path to a marimo .py file.
    """
    import marimo as mo

    graph = parse_notebook(notebook_path)
    diagram = render_mermaid(graph)
    n_cells = len(graph.cells)
    n_edges = len(graph.edges)
    n_md = sum(1 for c in graph.cells if c.kind == "md")
    n_imp = sum(1 for c in graph.cells if c.kind == "import")
    n_hidden = sum(1 for c in graph.cells if c.hide_code)
    header = mo.md(
        f"### Notebook DAG — `{Path(notebook_path).name}`\n\n"
        f"**{n_cells}** cells "
        f"({n_imp} import, {n_md} markdown, "
        f"{n_cells - n_md - n_imp} code) · "
        f"**{n_edges}** dependency edges · "
        f"**{n_hidden}** hide-code cells (marked `*`)"
    )
    return mo.vstack([header, mo.mermaid(diagram)])


# ---------------------------------------------------------------------------
# Internals


def _is_app_cell_decorator(decorators: list[ast.expr]) -> bool:
    for dec in decorators:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "app"
            and target.attr == "cell"
        ):
            return True
    return False


def _extract_return_names(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    for stmt in reversed(func.body):
        if not isinstance(stmt, ast.Return):
            continue
        if stmt.value is None:
            return ()
        if isinstance(stmt.value, ast.Tuple):
            return tuple(
                elt.id for elt in stmt.value.elts if isinstance(elt, ast.Name)
            )
        if isinstance(stmt.value, ast.Name):
            return (stmt.value.id,)
        return ()
    return ()


def _extract_label(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """First meaningful line of the cell, truncated to ~60 chars."""
    for stmt in func.body:
        if isinstance(stmt, ast.Return):
            continue
        md_text = _extract_md_text(stmt)
        if md_text is not None:
            return f"md: {md_text[:60]}"
        try:
            unparsed = ast.unparse(stmt).strip()
        except Exception:
            unparsed = "?"
        return unparsed.splitlines()[0][:60]
    return "(empty)"


def _extract_md_text(stmt: ast.stmt) -> str | None:
    """If stmt is `mo.md(...)`, return the first non-empty markdown line."""
    if not isinstance(stmt, ast.Expr):
        return None
    call = stmt.value
    if not isinstance(call, ast.Call):
        return None
    fn = call.func
    if not (isinstance(fn, ast.Attribute) and fn.attr in ("md", "markdown")):
        return None
    if not call.args:
        return None
    arg = call.args[0]

    raw: str | None = None
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        raw = arg.value
    elif isinstance(arg, ast.JoinedStr):
        for value in arg.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                raw = value.value
                break

    if raw is None:
        return ""

    for line in raw.splitlines():
        cleaned = line.strip().lstrip("# ").strip()
        if cleaned:
            return cleaned
    return ""


def _has_hide_code(decorators: list[ast.expr]) -> bool:
    for dec in decorators:
        if not isinstance(dec, ast.Call):
            continue
        for kw in dec.keywords:
            if kw.arg == "hide_code" and isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
    return False


def _classify(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Classify a cell as 'md', 'import', or 'code' for visual styling."""
    body = [s for s in func.body if not isinstance(s, ast.Return)]
    if not body:
        return "code"
    # md: first non-return is mo.md(...)
    if _extract_md_text(body[0]) is not None:
        return "md"
    # import: every non-return statement is an Import / ImportFrom
    if all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in body):
        return "import"
    return "code"


def _build_edges(cells: list[CellNode]) -> list[tuple[int, int]]:
    name_to_cell: dict[str, int] = {}
    for c in cells:
        for out in c.outputs:
            name_to_cell[out] = c.index  # last-write wins
    edges: list[tuple[int, int]] = []
    for c in cells:
        seen: set[int] = set()
        for inp in c.inputs:
            src = name_to_cell.get(inp)
            if src is None or src == c.index or src in seen:
                continue
            edges.append((src, c.index))
            seen.add(src)
    return edges


def _escape_mermaid_label(label: str) -> str:
    """Make a string safe to use as a mermaid node label."""
    return (
        label.replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("|", "/")
        .replace("\n", " ")
    )[:60]
