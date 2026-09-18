
# Code for Uniform Magnetic-Field 

This directory contains the code used to train, reconstruct, and analyze the uniform magnetic field using a physics-informed neural network (PINN) and spherical harmonics (SH).
## Files

- `physics_informed_neural_network.py`: Trains the PINN, reconstructs the magnetic field, and computes the field gradients and Maxwell-equation metrics.
- `spherical_harmonics.py`: Fits the measurement data using spherical harmonics, computes the \(g_{lm}\) coefficients, and reconstructs the magnetic field.
- **`uniform_field_analysis.ipynb` (main analysis notebook):** Runs the PINN and SH analyses, compares their results, and generates the figures and tables.

> **Note:** `uniform_field_analysis.ipynb` automatically uses the supporting custom modules `physics_informed_neural_network.py` and `spherical_harmonics.py`


## Running the analysis
- Clone or download the repository:
-  Specify the location of the repository on your computer:

```python
from pathlib import Path

project_dir = Path("/path/to/PINN-Magnetic-Field-Reconstruction")
```


- Run all notebook cells in order.

## PINN reconstruction

The PINN can be trained using 135, 540, or 1080 measurement points:

The function returns the trained model results, including:

- Reconstructed magnetic-field values
- Training-loss history
- First-order magnetic-field gradients
- Residual statistics
- Divergence and curl metrics

## Spherical-harmonic reconstruction

The spherical-harmonic model can be run as follows:

```python
sh_prediction_540_df, sh_glm_540_df, sh_linear_gradient_540_df = (
    sh.spherical_harmonics(540, l_max=3)
)
```

Here, `540` is the number of measurement points and `l_max=3` is the maximum degree of the spherical-harmonic expansion. Replace `540` with `135` or `1080` to use another measurement configuration.

## Figures and tables 

The analysis notebook generates:

- Mapper geometry (number of points) plots
- Simulated and PINN-reconstructed field maps at z=0 for Bz
- Field residual maps and distributions
- PINN training loss curves
- PINN and SH gradient comparisons
- Maxwell-equation metrics for divergence and curl
