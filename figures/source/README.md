# Figure 1

The editable vector layout and region annotations generate all four Figure 1 formats.
Use Python 3.10 or later with the figure-specific dependencies below.

```bash
python -m pip install -r figures/source/requirements.txt
python figures/source/build_figure1.py
```

The default font directory is Windows Fonts. On other systems, pass `--font-dir`
with a directory containing `arial.ttf`, `arialbd.ttf`, and `ariali.ttf`.
Use `--out` to write the SVG, PDF, PNG, and TIFF to another directory.

Positions are 1-based and inclusive in the 23-nt target DNA sequence. The seed
spans positions 13–20 and the PAM-proximal region spans 17–23, including the PAM.
The extended seed (13–23), used by FM6, is defined in Supplementary Table S5.
The FM12 example is the implemented product of seed and 5′-region melting-temperature
proxies. Figure generation does not change feature extraction or model predictions.
