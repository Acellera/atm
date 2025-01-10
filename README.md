# Acellera-ATM

Acellera-ATM is an [Alchemical Transfer Method (ATM)](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01129)
package for the Acellera platform.
The implementation is based on the AToM-OpenMM codebase https://github.com/Gallicchio-Lab/AToM-OpenMM
by Emilio Gallicchio's lab.

## Installation and usage

You can install the Acellera-ATM package via `conda` or `pip`. To use `conda` you need a working conda installation, we recommend using the
[Miniforge package](https://github.com/conda-forge/miniforge) for this.

Once conda is installed, you can install the Acellera-ATM package with the following command:

```bash
conda create -n atm -y # Create a new conda environment
conda activate atm     # Activate the environment
conda install acellera-atm python=3.10 -c acellera -c conda-forge # Install the package
```

## Usage

To use the Acellera-ATM package, you can either use it from command line or from python.
You can follow the tutorial [here](https://github.com/Acellera/quantumbind_rbfe) to get started.

### Citations

- [Relative Binding Free Energy Calculations for Ligands with Diverse Scaffolds with the Alchemical Transfer Method](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01129)

- [Alchemical Transfer Approach to Absolute Binding Free Energy Estimation](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00266)

- [Asynchronous Replica Exchange Software for Grid and Heterogeneous Computing](http://www.compmolbiophysbc.org/publications#asyncre_software_2015)
