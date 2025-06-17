# AIND Ophys NWB

A Python-based tool for working with ophys data in the Neurodata Without Borders (NWB) format. This library provides functionality to load and examine NWB files containing optical physiology data.

## Overview

This repository contains utilities for working with ophys (optical physiology) data stored in the NWB format. It allows users to load, examine, and process NWB files that contain multiplane ophys recordings. It is meant to be used in a Code Ocean workstation where processed ophys assets can be attached to the capsule and a reproducible run will produce a complete, ophys NWB.

## Usage

Loading NWB Files

The library provides functions to load NWB files from data directories:


## Examining NWB Content 

Once loaded, you can examine the NWB file structure and content using standard pynwb methods.

Example Notebooks

The repository includes example notebooks demonstrating typical workflows:

URL: https://github.com/AllenNeuralDynamics/aind-ophys-nwb/code/nwb_visualization_notebook.ipynb

File Path: /code/nwb_visualization_notebook.ipynb


## Data Structure

The library expects NWB files to be organized in a specific directory structure under `/data/` with folders containing "nwb" in their names. Raw data are expected to be mounted to a folder named `/raw`, the NWB schema must be mounted to `/schemas` and all processed data must be mounted to `/processed`.

## CodeOcean Integration

This repository is designed to work with CodeOcean, allowing for reproducible analysis of ophys data. The included capsule configuration provides all necessary dependencies.

## License

See the [LICENSE](LICENSE) file for details.

## Authors


Developed by the Allen Institute for Neural Dynamics (AIND) team.
