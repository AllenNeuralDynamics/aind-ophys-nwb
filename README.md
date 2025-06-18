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

Once loaded, you can examine the NWB file structure and content using standard pynwb methods.

Example Notebooks

The repository includes example notebooks demonstrating typical workflows:

URL: https://github.com/AllenNeuralDynamics/aind-ophys-nwb/code/nwb_visualization_notebook.ipynb

File Path: /code/nwb_visualization_notebook.ipynb


## CodeOcean Integration

This repository is designed to work with CodeOcean, allowing for reproducible analysis of ophys data. The included capsule configuration provides all necessary dependencies.

## License

See the [LICENSE](LICENSE) file for details.

## Authors


Developed by the Allen Institute for Neural Dynamics (AIND) team.
