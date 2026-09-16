# PINN Magnetic Field Reconstruction

This repository contains the data and code associated with the manuscript:

**"Reconstruction of Magnetic Fields with a Physics-Informed Neural Network"**

## Authors

P. Adhikari et al.  
Department of Physics and Astronomy  
University of Kentucky, Lexington, KY 40506, USA

## Overview

This work uses a physics-informed neural network (PINN) to reconstruct continuous three-dimensional magnetic fields and their spatial derivatives from discrete magnetic-field data.
Physical constraints are incorporated into the PINN through the source-free magnetostatic Maxwell equations,

$$
\nabla \cdot \mathbf{B} = 0,
$$

and

$$\nabla \times \mathbf{B} = 0.$$

The PINN reconstruction is demonstrated using two significantly different magnetic-field profiles: a highly uniform simulated magnetic field and a
rapidly varying experimentally measured magnetic field from the Spin-Transport Coils (STCs).

For the highly uniform magnetic field, the PINN reconstruction is also compared with a spherical-harmonic (SH) reconstruction.

## Repository Structure

- `data/` — datasets used for magnetic-field reconstruction and analysis
- `code/` — code used for PINN training, reconstruction, and analysis

## Data

The repository contains datasets supporting the magnetic-field
reconstruction examples presented in the manuscript.

The datasets include:

- Simulated data for a highly uniform magnetic field
- Experimental data for the rapidly varying Spin-Transport Coil (STC) field
- Measurement-point data used for reconstruction and validation

Magnetic-field datasets contain spatial coordinates and magnetic-field
components as applicable. Additional information about individual datasets,
including units and column definitions, is provided with the corresponding
data files.

## Code

The repository contains code used to perform the magnetic-field
reconstruction and analysis presented in the manuscript.

The code includes:

- Physics-informed neural-network (PINN) training and reconstruction
- Evaluation of the Maxwell-equation constraints
  $\nabla \cdot \mathbf{B}$ and $\nabla \times \mathbf{B}$
- Calculation of magnetic-field spatial derivatives
- Spherical-harmonic reconstruction and comparison
- Analysis and visualization of reconstructed magnetic fields

## Requirements

The analysis was performed in Python using packages including:

- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib

Detailed package dependencies and versions can be provided in a
`requirements.txt` file.

## Usage

Instructions for reproducing the magnetic-field reconstructions and
analysis are provided with the corresponding code.

## Citation

If you use the data or code from this repository, please cite the associated manuscript.

Full citation information will be added following publication.

## Data and Code Availability

The data and code supporting the results presented in the manuscript are available through this repository.

A permanent archival DOI will be provided through Zenodo.

## License

License information will be added to this repository.
