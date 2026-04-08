# Project Guidelines

## Code Style
- This workspace is primarily a LaTeX writing project under `repo/proposal/`.
- Keep edits minimal and preserve existing section structure and comments in `.tex` files.
- Use UTF-8 text and keep LaTeX package declarations centralized in `repo/proposal/proposal.tex`.
- When adding references, update `repo/proposal/reference.bib` and use `\cite{...}` keys consistently.

## Architecture
- Main entrypoint: `repo/proposal/proposal.tex`.
- Document composition uses `\input{...}` split files:
  - `repo/proposal/1-overview.tex`
  - `repo/proposal/2-formulation.tex`
  - `repo/proposal/3-evaluation.tex`
  - `repo/proposal/4-method.tex`
  - `repo/proposal/5-exp.tex`
  - `repo/proposal/6-contribution.tex`
- Style and submission format are controlled by `repo/proposal/neurips_2020.sty`.

## Build And Test
- Build from `repo/proposal/`.
- Preferred full build: `latexmk -pdf proposal.tex`.
- If bibliography changed, ensure BibTeX runs (latexmk handles this automatically).
- Clean generated files when needed: `latexmk -c`.

## Conventions
- Keep section ownership clear: add content to the corresponding numbered section file instead of putting body content directly in `proposal.tex`.
- Do not rename numbered section files unless all matching `\input{...}` references are updated.
- Preserve NeurIPS style options in `proposal.tex` unless the task explicitly asks to switch modes (e.g., `preprint` to `final`).
- Treat generated artifacts (`*.aux`, `*.bbl`, `*.fdb_latexmk`, `*.fls`) as build outputs; avoid manual edits.
