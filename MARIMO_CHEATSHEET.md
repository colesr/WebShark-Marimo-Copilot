# Marimo CLI Cheat Sheet

Quick reference for driving marimo from the terminal in this project.

## The one rule

Marimo isn't on PATH. **Every command starts with `.\.venv\Scripts\marimo.exe`** (relative to the project root), OR activate the venv once and use `marimo` directly.

```powershell
cd C:\Users\colec\PycharmProjects\DataScientist.ai
```

Then either prefix every call:

```powershell
.\.venv\Scripts\marimo.exe <subcommand> ...
```

Or activate the venv once and drop the prefix for the rest of the shell session:

```powershell
.\.venv\Scripts\Activate.ps1
marimo <subcommand> ...
```

The two are equivalent. Activation just trades one-time setup for shorter commands.

## Daily startup

```powershell
cd C:\Users\colec\PycharmProjects\DataScientist.ai
.\.venv\Scripts\marimo.exe edit notebooks\my-thing.py --no-token
```

Creates `my-thing.py` if missing, prints `http://localhost:2718`, opens it in a browser. **Use `--no-token`** so the marimo-pair Claude Code skill can discover the session.

## Core commands

| What you want                              | Command (after `cd` to project root)                                          |
|--------------------------------------------|-------------------------------------------------------------------------------|
| Edit a notebook (create if missing)        | `.\.venv\Scripts\marimo.exe edit notebooks\foo.py --no-token`                 |
| Start with NO browser auto-open            | `.\.venv\Scripts\marimo.exe edit notebooks\foo.py --no-token --headless`      |
| Run a finished notebook as a read-only app | `.\.venv\Scripts\marimo.exe run notebooks\foo.py`                             |
| Create a fresh empty notebook              | `.\.venv\Scripts\marimo.exe new`                                              |
| Convert a Jupyter `.ipynb` to marimo       | `.\.venv\Scripts\marimo.exe convert notebook.ipynb -o notebooks\notebook.py`  |
| Export to HTML                             | `.\.venv\Scripts\marimo.exe export html notebooks\foo.py -o foo.html`         |
| Walk the intro tutorial                    | `.\.venv\Scripts\marimo.exe tutorial intro`                                   |
| Check available tutorials                  | `.\.venv\Scripts\marimo.exe tutorial --help`                                  |
| Print marimo env info (versions, paths)    | `.\.venv\Scripts\marimo.exe env`                                              |
| Check / format a marimo file               | `.\.venv\Scripts\marimo.exe check notebooks\foo.py`                           |

## Useful flags

| Flag             | Effect                                                                        |
|------------------|-------------------------------------------------------------------------------|
| `--no-token`     | No auth token. Required for marimo-pair auto-discovery. Local-dev only.       |
| `--headless`     | Don't auto-open a browser tab. Useful when you'll open the URL manually.      |
| `--port 2719`    | Use a non-default port. Default is 2718.                                      |
| `--watch`        | Reload the kernel when the `.py` file changes on disk. Off by default.        |
| `--sandbox`      | Run in an isolated environment (uv-managed). Bypass project deps.             |
| `-h` / `--help`  | Show help for marimo OR for any subcommand (e.g. `edit --help`).              |

## Stopping marimo

- **From the terminal that launched it**: `Ctrl+C`. Cleanly shuts down.
- **If it's running in the background**: close the terminal window, OR find the PID and `taskkill /PID <pid> /F`.
- **Just close the browser tab**: the kernel keeps running. Doesn't stop the process.

## Running pytest (same venv, not marimo-specific but related)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Or with the venv activated, just `pytest tests/ -v`.

## Common gotchas

- **`ModuleNotFoundError: No module named 'ds_copilot'`**: you launched marimo with the wrong Python. Always use `.\.venv\Scripts\marimo.exe`, not just `marimo`.
- **"Port 2718 already in use"**: a marimo session is still running. Either reuse the tab in the browser, or kill the old process before starting a new one.
- **Notebook outside the project dir**: still works *if* you point at the project venv explicitly:
  ```powershell
  C:\Users\colec\PycharmProjects\DataScientist.ai\.venv\Scripts\marimo.exe edit C:\path\to\file.py --no-token
  ```
  But `.ds_copilot/decisions.jsonl` lands in whatever directory you're `cd`'d into. Usually easier to keep notebooks under the project.

## Verify the venv is wired up

Paste this into any new marimo cell:

```python
import ds_copilot
print(ds_copilot.__version__)
```

Should print `0.0.1`. If it errors, the venv is wrong — restart marimo with `.\.venv\Scripts\marimo.exe`.
