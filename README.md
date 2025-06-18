# AIND Ophys NWB

A Python-based tool for working with ophys data in the Neurodata Without Borders (NWB) format. This library provides functionality to load and examine NWB files containing optical physiology data.

## Overview

This repository extracts processed data from single plane and multiplane optical physiology datasets and packages the data into NWB format.It is meant to be used in a Code Ocean workstation where processed ophys assets can be attached to the capsule and a reproducible run will produce a complete, ophys NWB.  It also containes notebooks which allow users to load, examine, and process NWB files that contain multiplane ophys recordings.

## Usage

Loading NWB Files
The library provides functions to load NWB files from data directories.
In Code Ocean, the required input are:
1. NWB files to be organized in a specific directory structure under `/data/` with folders containing "nwb" in their names.
2. Raw data are expected to be mounted to a folder named `/raw`
3. NWB schema must be mounted to `/schemas`
4. All processed data must be mounted to `/processed`.


## Examining NWB Content 

Provided is an [example notebook]([https://github.com/AllenNeuralDynamics/aind-ophys-nwb/examples/nwb_visualization_notebook.ipynb](https://github.com/AllenNeuralDynamics/aind-ophys-nwb/blob/main/examples/nwb_visualization_notebook.ipynb)) examining the contents of the NWB.

Briefly, the contents output in the NWB include: 

1) Processing container with the cell table for each plane. The cell table includes, extraction traces, dF / F, segmentation and cell classification.

2) Subject container with the subject data.

3) Device container with the microscope information


## CodeOcean Integration

This repository is designed to work with CodeOcean, allowing for reproducible analysis of ophys data. The included capsule configuration provides all necessary dependencies.

## License

See the [LICENSE](LICENSE) file for details.

## Authors


Developed by the Allen Institute for Neural Dynamics (AIND) team.
