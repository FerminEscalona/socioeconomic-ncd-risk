# Repository Guidelines

## Project Structure & Module Organization

This repository is in an early planning phase for a global health and socioeconomic risk analysis project. Current files are:

- `README.md`: Spanish-language project overview, research question, source context, and intended scope. Understand it.
- `working/`: draft planning documents and implementation notes, such as `PRD_01_data_extraction.md`.

When source code is introduced, keep reusable logic under a clear source directory such as `src/`, exploratory notebooks under `notebooks/`, raw immutable inputs under `data/raw/`, cleaned outputs under `data/processed/`, and tests under `tests/`. Do not commit large downloaded datasets unless the project explicitly adopts data versioning.

## Build, Test, and Development Commands

No build system, runtime manifest, or test runner is currently defined. Until one is added, use repository inspection commands only:

- `git status --short`: check local changes before editing.
- `rg --files`: list tracked and untracked project files quickly.
- `sed -n '1,160p' README.md`: review project context without opening an editor.

When adding Python tooling, prefer documenting commands in a `Makefile` or `pyproject.toml`, for example `make test`, `make lint`, and `make format`.

## Coding Style & Naming Conventions

Keep Markdown concise and structured with sentence-case headings. The README is currently Spanish; preserve Spanish for project-facing documentation unless a new file is explicitly intended for English-speaking contributors.

For future Python code, use 4-space indentation, descriptive snake_case names for modules/functions, PascalCase for classes, and uppercase names for constants. Prefer explicit names tied to the domain, for example `ncd_mortality_rate`, `world_bank_indicator`, and `country_year_panel`.

### Code Simplicity & Documentation (MANDATORY)

- All code must prioritize **simplicity over complexity**.
- Avoid over-engineering, unnecessary abstractions, or premature optimization.
- Prefer clear and readable logic even if it is slightly more verbose.
- Each logical block ("chunk") of code must include **clear comments explaining**:
  - What it does
  - Why it is needed
- Comments should be concise but meaningful, enabling quick understanding by any contributor.

## Testing Guidelines

There are no tests yet. Add tests alongside the first source modules under `tests/`, using filenames like `test_data_extraction.py` or `test_indicator_mapping.py`. Focus initial coverage on data cleaning rules, indicator transformations, schema validation, and API response parsing. Tests should run without network access by using fixtures or cached sample responses.

## Commit & Pull Request Guidelines

Existing history uses short, imperative commit messages, for example `Initialize README with project description and goals`. Continue that style: describe the change, not the process.

Pull requests should include a concise summary, affected files or data sources, validation performed, and any known limitations. For analysis or visualization changes, include screenshots, sample outputs, or links to generated artifacts when relevant.

## Security & Configuration Tips

Do not commit API keys, credentials, downloaded private data, or local environment files. Keep source URLs, indicator IDs, and reproducibility notes in documented configuration files rather than hard-coded notebook cells.