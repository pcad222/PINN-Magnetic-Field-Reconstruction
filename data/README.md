# Data

This directory contains the datasets used for the magnetic field reconstruction and analysis presented in the manuscript:

**"Reconstruction of Magnetic Fields with a Physics-Informed Neural Network"**

The data consist of two different magnetic field profiles:
- a highly uniform simulated magnetic field and a rapidly varying
- experimentally measured magnetic field from the Spin-Transport Coils (STCs).

## Highly Uniform Magnetic Field

The simulated highly uniform magnetic field was sampled using three measurement point configurations:

- `uniform_field_135_points.csv` — 135 measurement points
- `uniform_field_540_points.csv` — 540 measurement points
- `uniform_field_1080_points.csv` — 1080 measurement points



Each dataset contains the spatial coordinates `x`, `y`, and `z` in meters
(m) and the corresponding magnetic field components `bx`, `by`, and `bz`
in nanotesla (nT).



## Rapidly Varying Spin-Transport Coil (STC) Field

The experimentally measured rapidly varying magnetic field consists of measurements from four Spin-Transport Coils (STCs), labeled A–D.

The datasets include:

- `SUA_10mA_subtracted.txt` 
- `SUA_20mA_subtracted.txt` 
- `SUB_50mA_subtracted.txt` 
- `SUB_100mA_subtracted.txt` 
- `SUC_50mA_subtracted.txt` 
- `SUC_100mA_subtracted.txt` 
- `SUD_50mA_subtracted.txt` 
- `SUD_100mA_subtracted.txt` 

The STC text files contain multiple columns. For the magnetic field reconstruction presented in the manuscript, the first seven columns are
used in the following order:

`x`, `y`, `z`, `i`, `bx`, `by`, `bz`

The spatial coordinates `x`, `y`, and `z` are given in centimeters (cm), and the magnetic field components `bx`, `by`, and `bz` are given in
microtesla (µT).


The STC data are read in the analysis using:

```python
df = pd.read_csv(file, header=None, sep=r"\s+").iloc[:, :7]
df.columns = ["x", "y", "z", "i", "bx", "by", "bz"]
```

## Directory Structure

```text
data/
├── README.md
├── uniform_field/
│   ├── uniform_field_135_points.csv
│   ├── uniform_field_540_points.csv
│   └── uniform_field_1080_points.csv
│
└── stc_field/
    ├── SUA_10mA_subtracted.txt
    ├── SUA_20mA_subtracted.txt
    ├── SUB_50mA_subtracted.txt
    ├── SUB_100mA_subtracted.txt
    ├── SUC_50mA_subtracted.txt
    ├── SUC_100mA_subtracted.txt
    ├── SUD_50mA_subtracted.txt
    └── SUD_100mA_subtracted.txt
```

These datasets support the magnetic field reconstruction, validation,
and comparison studies described in the manuscript.
