# PINN Magnetic Field Reconstruction

This repository contains the data and code associated with the manuscript:

**"Reconstruction of Magnetic Fields with a Physics-Informed Neural Network"**

## Authors

P. Adhikari et al.  
Department of Physics and Astronomy, University of Kentucky, Lexington, KY 40506, USA

## Overview

This work uses a physics-informed neural network (PINN) to reconstruct continuous three-dimensional magnetic fields and their spatial derivatives
from discrete magnetic field data. Physics constraints are incorporated into the PINN through the source-free
magnetostatic Maxwell equations,

$$
\nabla \cdot \mathbf{B} = 0
\quad \text{and} \quad
\nabla \times \mathbf{B} = 0.
$$

The method is demonstrated using two significantly different magnetic field profiles: a highly uniform simulated magnetic field and a rapidly varying
experimentally measured magnetic field from the Spin-Transport Coils (STCs).

For the highly uniform magnetic field, the reconstructed field is compared with the simulated reference field through residual distributions. The performance of the PINN is also benchmarked against a spherical-harmonic (SH) reconstruction by comparing the reconstructed fields and their first-order spatial gradients. The effect of varying the number of
measurement points on the reconstruction is also investigated.

For the rapidly varying STC field, the PINN is applied to experimental measurements to reconstruct the magnetic field profile.

## Repository Structure

- `data/` — datasets used for magnetic field reconstruction and analysis
- `code/` — code used for PINN training, reconstruction, and analysis

## Data

The repository contains datasets supporting the magnetic field reconstruction presented in the manuscript.

The datasets include:

- Simulated data for a highly uniform magnetic field
- Experimental data for the rapidly varying Spin Transport Coil (STC) field

The magnetic field datasets contain the spatial coordinates `x`, `y`, and `z` and the corresponding magnetic field components `Bx`, `By`, and `Bz`.

For the highly uniform magnetic field dataset, the spatial coordinates `x`, `y`, and `z` are given in meters (m), and the magnetic field components `Bx`, `By`, and `Bz` are given in nanotesla (nT). For the STC dataset, the spatial coordinates `x`, `y`, and `z` are given in centimeters (cm), and the magnetic field components `Bx`, `By`, and `Bz` are given in microtesla (µT).

## Code

The repository contains code used to perform the magnetic field reconstruction and analysis presented in the manuscript.

The code includes:

- Physics-informed neural network (PINN) training and reconstruction
- Evaluation of the Maxwell-equation constraints
  $\nabla \cdot \mathbf{B}$ and $\nabla \times \mathbf{B}$
- Calculation of magnetic-field spatial derivatives
- Spherical-harmonic reconstruction and comparison
  
## Requirements

The analysis was performed in Python using packages including:

- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib

## Usage

Instructions for reproducing the magnetic field reconstructions and analysis are provided with the corresponding code.

## Citation

If you use the data or code from this repository, please cite the associated manuscript.

## Data and Code Availability

The data and code supporting the results presented in the manuscript are available through this repository.

## License

License information will be added to this repository.
