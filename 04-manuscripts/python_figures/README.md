# Manuscript Figure Rebuild (Python)

This folder contains a Python driver to regenerate manuscript figures from:

- `FM5_ML` data (`02-data`, `04-manuscripts`)
- sibling repos used by the original MATLAB workflow (`FM`, `FM3`, `FM4`)

## Files

- `make_manuscript_figures.py`: builds Figures 1-14.
- `requirements.txt`: minimal Python dependencies.

## Quick Start

From repository root:

```bash
python3 04-manuscripts/python_figures/make_manuscript_figures.py --figures all
```

Output is saved to:

`04-manuscripts/python_figures/output/`

## Useful Options

- Build specific figures:

```bash
python3 04-manuscripts/python_figures/make_manuscript_figures.py --figures 1,2,3,10,11
```

- Override sibling repo locations:

```bash
python3 04-manuscripts/python_figures/make_manuscript_figures.py \
  --fm-root /path/to/FM \
  --fm3-root /path/to/FM3 \
  --fm4-root /path/to/FM4
```

- If a figure fails, fall back to extracting `imageN.png` from manuscript DOCX:

```bash
python3 04-manuscripts/python_figures/make_manuscript_figures.py \
  --figures all \
  --fallback-docx-image
```

## Notes

- Figure 12 beachball-style symbols are generated from catalog vectors (`avfnorm`, `avslip`) using a Python port of the MATLAB plotting logic.
- Figure 13 Kagan angles are computed with a Python port of the MATLAB quaternion-based implementation.
