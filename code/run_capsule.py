import argparse
import json
import glob
import os
import logging
import fnmatch
import pandas as pd
import imageio
import shutil
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union
import re

# capsule
import file_handling
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import sparse
from aind_metadata_mapper.open_ephys.utils import sync_utils as sync
from hdmf.common import VectorData
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from pynwb.image import GrayscaleImage, Images, ImageSeries
from pynwb.ophys import (
    DfOverF,
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
    RoiResponseSeries,
)
from pynwb.epoch import TimeIntervals
from schemas import OphysMetadata


class SegmentationApproach(Enum):
    SUITE2P_ANATOMICAL = {
        "method": "suite2p-cellpose",
        "description": "Suite2p's anatomical mode using either "
        "the max or mean projection",
    }
    SUITE2P_ACTIVITY = {
        "method": "suite2p-functional",
        "description": "Suite2p's activity-based ROI detection using "
        "'sparse mode''",
    }


def load_pynwb_extension(schema, path):
    neurodata_type = "OphysMetadataSchema"
    pynwb.load_namespaces(path)
    return pynwb.get_class(neurodata_type, "ndx-aibs-behavior-ophys")


def grab_suffixes(directory):
    """
    Finds unique suffixes from CSV files and checks if
    a corresponding TIFF file exists.

    :param directory: Path to the directory containing .csv and .tif files.
    :return: A set of valid suffixes that have both .csv and .tif files.
    """
    csv_suffixes = set()
    tiff_suffixes = set()

    # Iterate through files and collect suffixes
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            suffix = filename.split("_", 1)[0] + "_"
            csv_suffixes.add(suffix)
        elif filename.endswith(".tif"):
            suffix = filename.split("_", 1)[0] + "_"
            tiff_suffixes.add(suffix)

    # Find suffixes that have both CSV and TIFF files
    valid_suffixes = csv_suffixes & tiff_suffixes

    return valid_suffixes


def add_tiffs_to_nwb(directory, suffix, nwb_file):
    """
    Adds specific TIFF frames as an ImageSeries to an
      NWB file based on frame numbers in an associated CSV.

    :param directory: Directory containing the TIFF and CSV files.
    :param suffix: Suffix to match TIFF and CSV files.
    :param nwb_file: The NWB file object to which the
                     ImageSeries will be added.
    :return: The updated NWB file.
    """

    print(f"Processing suffix: {suffix}")

    # Get all .tif files matching the suffix
    tiff_files = sorted(
        [
            f
            for f in os.listdir(directory)
            if fnmatch.fnmatch(f, f"*{suffix}*.tif")
        ]
    )

    if not tiff_files:
        raise ValueError(
            f"No TIFF files found with suffix '{suffix}' in {directory}"
        )

    # Find the first associated CSV file
    csv_files = sorted(
        [
            f
            for f in os.listdir(directory)
            if fnmatch.fnmatch(f, f"*{suffix}*.csv")
        ]
    )

    if not csv_files:
        raise ValueError(
            f"No CSV file found for suffix '{suffix}' in {directory}"
        )

    csv_path = os.path.join(
        directory, csv_files[0]
    )  # Use the first CSV file found

    # Load frame numbers and timestamps from CSV
    df = pd.read_csv(
        csv_path
    )  # Assuming CSV has "frameNumber" and "timestamp" columns

    if "frameNumber" not in df.columns or "timestamp" not in df.columns:
        return nwb_file

    valid_frames = set(df["frameNumber"])  # The frames we want to extract
    frame_to_time = dict(
        zip(df["frameNumber"], df["timestamp"])
    )  # Map frameNumber to timestamp

    images = []
    timestamps = []

    # Read all images but only store the ones in 'valid_frames'
    for filename in tiff_files:
        filepath = os.path.join(directory, filename)
        reader = imageio.get_reader(filepath)  # Open TIFF stack reader

        for frame_index, frame in enumerate(reader):
            if frame_index in valid_frames:  # Only keep frames from CSV
                images.append(frame)
                timestamps.append(
                    frame_to_time[frame_index]
                )  # Assign timestamp

    if not images:
        raise ValueError("No matching frames found in TIFF files.")

    # Store images as a single ImageSeries with timestamps
    img_series = ImageSeries(
        name=f"image_series_{suffix}",
        data=images,  # Store images as a list
        unit="pixels",
        format="raw",
        timestamps=timestamps,  # Assign timestamps from CSV
    )

    # Add ImageSeries to NWB
    nwb_file.add_acquisition(img_series)
    return nwb_file


def load_signals(
    h5_file: Path,
    plane_segmentation: ImageSegmentation,
    h5_group=None,
    h5_key=None,
) -> tuple:
    """Loads extracted signal data from aind-ophys-extraction-suite2p

    Parameters
    ----------
    h5_file: Path
        Path to h5_file
    ps: ImageSegmentation
        nwb segmented object
    h5_group: str
        Group to access key in h5 file
    h5_key: str
        Key to extract data from

    Returns
    -------
    (np.array, ImageSegmentation)
        Trace array and updated segmentation object
    """
    if not h5_group:
        with h5py.File(h5_file, "r") as f:
            traces = f[h5_key][:]
    else:
        with h5py.File(h5_file, "r") as f:
            traces = f[h5_group][h5_key][:]
    index = traces.shape[0]

    roi_names = np.arange(index).tolist()
    rt_region = plane_segmentation.create_roi_table_region(
        region=roi_names, description="List of measured ROIs"
    )

    return traces, rt_region


def load_generic_group(h5_file: Path, h5_group=None, h5_key=None) -> np.array:
    """Loads extracted signal data from aind-ophys-extraction-suite2p

    Parameters
    ----------
    h5_file: Path
        Path to h5_file
    h5_group: str
        Group to access key in h5 file
    h5_key: str
        Key to extract data from

    Returns
    -------
    (np.array)
        data array
    """
    if h5_group:
        with h5py.File(h5_file, "r") as f:
            return f[h5_group][h5_key][:]
    else:
        with h5py.File(h5_file, "r") as f:
            return f[h5_key][:]


def load_sparse_array(h5_file):
    """Obtain pixel masks from the h5 file

    Parameters
    ----------
    h5_file : Path
        The path to the h5 file

    Returns
    -------
    np.array
        The pixel masks
    """
    with h5py.File(h5_file) as f:
        data = f["rois"]["data"][:]
        coords = f["rois"]["coords"][:]
        shape = f["rois"]["shape"][:]

    pixelmasks = sparse.COO(coords, data, shape).todense()
    return pixelmasks


def get_segementation_approach(extraction_h5: Path) -> SegmentationApproach:
    """Get the segmentation approach from the extraction file

    Parameters
    ----------
    extraction_h5 : Path
        The path to the extraction file

    Returns
    -------
    SegmentationApproach
        The segmentation approach, either SUITE2P_ANATOMICAL
        or SUITE2P_ACTIVITY
    """
    with h5py.File(extraction_h5, "r") as f:
        if f.get("cellpose", False):
            return SegmentationApproach.SUITE2P_ANATOMICAL
        else:
            return SegmentationApproach.SUITE2P_ACTIVITY


def get_microscope(
    nwbfile: pynwb.NWBFile, rig_json_data: dict, session_json_data: dict,
) -> Tuple[pynwb.device.Device, OpticalChannel]:
    """Get microscope metadata for the NWB file

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file
    rig_json_data : dict
        The rig metadata
    session_json_data : dict
        The session metadata

    Returns
    -------
    Tuple[pynwb.device, OpticalChannel]
        The device and optical channel
    """
    microscope_name = session_json_data["rig_id"]
    microscope_desc = rig_json_data["rig_id"]
    microscope_manufacturer = "Thorlabs"  # TODO UPDATE IN rig.json
    # optical channels
    oc1_name = "TODO"
    oc1_desc = "TODO"
    oc1_el = 514.0  # TODO placeholder on Gcamp6 for now emission lambda
    device = nwbfile.create_device(
        name=microscope_name,
        description=microscope_desc,
        manufacturer=microscope_manufacturer,
    )
    optical_channel = OpticalChannel(
        name=oc1_name, description=oc1_desc, emission_lambda=oc1_el,
    )
    return device, optical_channel


def nwb_ophys_single_plane(
    nwbfile: pynwb.NWBFile,
    file_paths: dict,
    rig_json_data: dict,
    session_json_data: dict,
    subject_json_data: dict,
    frame_rate: float,
) -> Tuple[pynwb.NWBFile, dict]:
    """Create an NWB file for single-plane ophys data
       using frame rate instead of timestamps

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file
    file_paths : dict
        The paths to the processed files
    rig_json_data : dict
        The rig metadata
    session_json_data : dict
        The session metadata
    subject_json_data : dict
        The subject metadata
    frame_rate : float
        The frame rate in Hz (frames per second)

    Returns
    -------
    pynwb.NWBFile
        The NWB file
    dict
        The overall metadata
    """
    device, optical_channel = get_microscope(
        nwbfile, rig_json_data, session_json_data
    )

    # Get the single plane name from file_paths
    plane_name = list(file_paths["planes"].keys())[0]
    logging.info(f"Processing single plane: {plane_name}")

    # Create imaging plane with appropriate metadata
    location = (
        "Structure: " + "Unknown Structure" + " Depth: " + str(0)
    )  # Update if available in metadata

    ophys_module = nwbfile.create_processing_module(
        name=plane_name, description="Single-plane ophys processing module"
    )

    imaging_plane = nwbfile.create_imaging_plane(
        name=plane_name,
        optical_channel=optical_channel,
        imaging_rate=frame_rate,
        description="Single-photon imaging plane",
        device=device,
        excitation_lambda=float(
            session_json_data["data_streams"][0]["light_sources"][0][
                "wavelength"
            ]
        ),
        indicator=subject_json_data["genotype"],
        location=location,
        grid_spacing=[1.0, 1.0],  # Update if available
        grid_spacing_unit="meters",
        origin_coords=[0.0, 0.0, 0.0],
        origin_coords_unit="meters",
    )

    # Add ImageSegmentation
    img_seg = ImageSegmentation(name="image_segmentation")

    # Determine segmentation approach
    try:
        segmentation_approach = get_segementation_approach(
            file_paths["planes"][plane_name]["extraction_h5"]
        )
        plane_seg_approach = segmentation_approach.value["method"]
        plane_seg_descr = segmentation_approach.value["description"]
    except Exception as e:
        logging.warning(f"Could not determine segmentation approach: {e}")
        plane_seg_approach = "unknown"
        plane_seg_descr = "Unknown segmentation method"

    # Create plane segmentation
    try:
        # Try to load classifier data
        soma_predictions = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="soma",
            h5_key="predictions",
        )
        soma_probabilities = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="soma",
            h5_key="probabilities",
        )
        dendrite_predictions = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="dendrites",
            h5_key="predictions",
        )
        dendrite_probabilities = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="dendrites",
            h5_key="probabilities",
        )

        plane_segmentation = img_seg.create_plane_segmentation(
            name="roi_table",
            description=plane_seg_approach + plane_seg_descr,
            imaging_plane=imaging_plane,
            columns=[
                VectorData(name="is_soma", description="Soma predictions",),
                VectorData(
                    name="soma_probability", description="Soma probabilities",
                ),
                VectorData(
                    name="is_dendrite", description="Dendrite predictions",
                ),
                VectorData(
                    name="dendrite_probability",
                    description="Dendrite probabilities",
                ),
            ],
            colnames=[
                "is_soma",
                "soma_probability",
                "is_dendrite",
                "dendrite_probability",
            ],
        )

        for idx, pixel_mask in enumerate(
            load_sparse_array(
                file_paths["planes"][plane_name]["extraction_h5"]
            )
        ):

            plane_segmentation.add_roi(
                image_mask=pixel_mask,
                is_soma=soma_predictions[idx],
                soma_probability=soma_probabilities[idx][-1],
                is_dendrite=dendrite_predictions[idx],
                dendrite_probability=dendrite_probabilities[idx][-1],
            )
    except Exception as e:
        logging.warning(f"Error adding ROIs with classifier data: {e}")
        # Fallback to simpler plane segmentation without classifier data
        plane_segmentation = img_seg.create_plane_segmentation(
            name="roi_table",
            description=plane_seg_approach + plane_seg_descr,
            imaging_plane=imaging_plane,
        )
        for idx, pixel_mask in enumerate(
            load_sparse_array(
                file_paths["planes"][plane_name]["extraction_h5"]
            )
        ):

            plane_segmentation.add_roi(image_mask=pixel_mask)

    ophys_module.add(img_seg)

    # Add projections and images
    try:
        # Add average and max projections if available
        avg_projection = plt.imread(
            file_paths["planes"][plane_name]["average_projection_png"]
        )
        avg_img = GrayscaleImage(
            name="average_projection",
            data=avg_projection,
            resolution=1.0,  # Update if available
            description="Average intensity projection of entire session",
        )

        max_projection = plt.imread(
            file_paths["planes"][plane_name]["max_projection_png"]
        )
        max_img = GrayscaleImage(
            name="max_projection",
            data=max_projection,
            resolution=1.0,  # Update if available
            description="Max intensity projection of entire session",
        )

        # Try to add segmentation mask if available
        if segmentation_approach == SegmentationApproach.SUITE2P_ANATOMICAL:
            segmentation_mask = load_generic_group(
                file_paths["planes"][plane_name]["extraction_h5"],
                h5_group="cellpose",
                h5_key="masks",
            )
            mask_img = GrayscaleImage(
                name="segmentation_mask_image",
                data=segmentation_mask,
                resolution=1.0,  # Update if available
                description="Segmentation projection of entire session",
            )
            images = Images(
                name="images",
                images=[avg_img, max_img, mask_img],
                description="Summary images of the ophys movie",
            )
        else:
            images = Images(
                name="images",
                images=[avg_img, max_img],
                description="Summary images of the ophys movie",
            )

        ophys_module.add(images)
    except Exception as e:
        logging.warning(f"Error adding projection images: {e}")
    """
    if file_paths['planes'][plane_name]["classifier_h5"] is None:
        return nwbfile
    """
    # Add timeseries data
    try:
        # ROI traces
        roi_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="roi",
        )
        print(roi_traces, roi_names)

        # Create time series using frame rate
        roi_traces_series = RoiResponseSeries(
            name="ROI_fluorescence",
            data=roi_traces.T,
            rois=roi_names,
            unit="a.u.",
            rate=frame_rate,
            starting_time=0.0,
        )

        # Neuropil traces
        neuropil_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="neuropil",
        )

        neuropil_traces_series = RoiResponseSeries(
            name="neuropil_fluorescence",
            data=neuropil_traces.T,
            rois=roi_names,
            unit="a.u.",
            rate=frame_rate,
            starting_time=0.0,
        )

        # Neuropil corrected traces if available
        try:
            neuropil_corrected, roi_names = load_signals(
                file_paths["planes"][plane_name]["extraction_h5"],
                plane_segmentation,
                h5_group="traces",
                h5_key="corrected",
            )

            neuropil_corrected_series = RoiResponseSeries(
                name="neuropil_corrected",
                data=neuropil_corrected.T,
                rois=roi_names,
                unit="a.u.",
                rate=frame_rate,
                starting_time=0.0,
            )

            ophys_module.add(neuropil_corrected_series)
        except Exception as e:
            logging.warning(f"Error adding neuropil corrected traces: {e}")

        # DfOverF traces if available
        try:
            dfof_traces, roi_names = load_signals(
                file_paths["planes"][plane_name]["dff_h5"],
                plane_segmentation,
                h5_key="data",
            )

            dfof_traces_series = RoiResponseSeries(
                name="dff",
                data=dfof_traces.T,
                rois=roi_names,
                unit="%",
                rate=frame_rate,
                starting_time=0.0,
            )

            ophys_module.add(
                DfOverF(roi_response_series=dfof_traces_series, name="dff")
            )
        except Exception as e:
            logging.warning(f"Error adding DfOverF traces: {e}")

        # Event traces if available
        try:
            event_traces, roi_names = load_signals(
                file_paths["planes"][plane_name]["events_oasis_h5"],
                plane_segmentation,
                h5_key="events",
            )

            event_traces_series = RoiResponseSeries(
                name="event",
                data=event_traces.T,
                rois=roi_names,
                unit="a.u.",
                rate=frame_rate,
                starting_time=0.0,
            )

            ophys_module.add(event_traces_series)
        except Exception as e:
            logging.warning(f"Error adding event traces: {e}")

        # Add the base fluorescence data to the module
        ophys_module.add(
            Fluorescence(roi_response_series=roi_traces_series, name="raw")
        )
        ophys_module.add(neuropil_traces_series)

    except Exception as e:
        logging.error(f"Error adding timeseries data: {e}")
        raise

    return nwbfile


def nwb_ophys(
    nwbfile: pynwb.NWBFile,
    file_paths: dict,
    ophys_fovs: list,
    rig_json_data: dict,
    session_json_data: dict,
    subject_json_data: dict,
) -> Tuple[pynwb.NWBFile, dict]:
    """Create an NWB file for ophys data

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file
    file_paths : dict
        The paths to the processed files
    ophys_fovs : list
        The session metadata
    rig_json_data : dict
        The rig metadata
    session_json_data : dict
        The session metadata
    subject_json_data : dict
        The subject metadata
    processing_json_data : dict
        The processing metadata

    Returns
    -------
    pynwb.NWBFile
        The NWB file
    dict
        The overall metadata
    """
    device, optical_channel = get_microscope(
        nwbfile, rig_json_data, session_json_data
    )
    # Start plane specific processing
    for plane in ophys_fovs:
        plane_name = f"{plane['targeted_structure']}_{plane['index']}"
        logging.info(f"Processing plane: {plane_name}")
        location = (
            "Structure: "
            + plane["targeted_structure"]
            + " Depth: "
            + str(plane["imaging_depth"])
        )
        ophys_module = nwbfile.create_processing_module(
            name=plane_name, description=""
        )
        imaging_plane = nwbfile.create_imaging_plane(
            name=plane_name,  # ophys_plane_id
            optical_channel=optical_channel,
            imaging_rate=float(plane["frame_rate"]),
            description="Two-photon imaging plane a",
            device=device,
            excitation_lambda=float(
                session_json_data["data_streams"][0]["light_sources"][0][
                    "wavelength"
                ]
            ),
            indicator=subject_json_data["genotype"],
            location=location,
            grid_spacing=[
                float(plane["fov_scale_factor"]),
                float(plane["fov_scale_factor"]),
            ],
            grid_spacing_unit=plane["fov_coordinate_unit"],
            origin_coords=[0.0, 0.0, 0.0],  # TODO: dunno
            origin_coords_unit=plane["fov_coordinate_unit"],
        )
        segmentation_approach = get_segementation_approach(
            file_paths["planes"][plane_name]["extraction_h5"]
        )
        plane_seg_approach = segmentation_approach.value["method"]
        plane_seg_descr = segmentation_approach.value["description"]
        img_seg = ImageSegmentation(name="image_segmentation")
        soma_predictions = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="soma",
            h5_key="predictions",
        )
        soma_probabilities = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="soma",
            h5_key="probabilities",
        )
        dendrite_predictions = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="dendrites",
            h5_key="predictions",
        )
        dendrite_probabilities = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="dendrites",
            h5_key="probabilities",
        )
        plane_segmentation = img_seg.create_plane_segmentation(
            name="roi_table",
            description=plane_seg_approach + plane_seg_descr,
            imaging_plane=imaging_plane,
            columns=[
                VectorData(name="is_soma", description="Soma predictions",),
                VectorData(
                    name="soma_probability", description="Soma probabilities",
                ),
                VectorData(
                    name="is_dendrite", description="Dendrite predictions",
                ),
                VectorData(
                    name="dendrite_probability",
                    description="Dendrite probabilities",
                ),
            ],
            colnames=[
                "is_soma",
                "soma_probability",
                "is_dendrite",
                "dendrite_probability",
            ],
        )
        ophys_module.add(img_seg)

        avg_projection = plt.imread(
            file_paths["planes"][plane_name]["average_projection_png"]
        )
        avg_img = GrayscaleImage(
            name="average_projection",
            data=avg_projection,
            resolution=float(plane["fov_scale_factor"]),
            description="Average intensity projection of entire session",
        )
        max_projection = plt.imread(
            file_paths["planes"][plane_name]["max_projection_png"]
        )
        max_img = GrayscaleImage(
            name="max_projection",
            data=max_projection,
            resolution=float(plane["fov_scale_factor"]),  # pixels/cm
            description="Max intensity projection of entire session",
        )
        if segmentation_approach == SegmentationApproach.SUITE2P_ANATOMICAL:
            segmetation_mask = load_generic_group(
                file_paths["planes"][plane_name]["extraction_h5"],
                h5_group="cellpose",
                h5_key="masks",
            )
        else:
            raise NotImplementedError(
                "Cannot process functional segmentation"
            )
        mask_img = GrayscaleImage(
            name="segmentation_mask_image",
            data=segmetation_mask,
            resolution=float(plane["fov_scale_factor"]),  # pixels/cm
            description="Segmentation projection of entire session",
        )

        images = Images(
            name="images",
            images=[avg_img, max_img, mask_img],
            description="Summary images of the two-photon movie",
        )
        ophys_module.add(images)

        # 4D. ROIS

        rois_shape = load_generic_group(
            file_paths["planes"][plane_name]["extraction_h5"],
            h5_group="rois",
            h5_key="shape",
        )
        for idx, pixel_mask in enumerate(
            load_sparse_array(
                file_paths["planes"][plane_name]["extraction_h5"]
            )
        ):
            plane_segmentation.add_roi(
                image_mask=pixel_mask,
                is_soma=soma_predictions[idx],
                soma_probability=soma_probabilities[idx][
                    -1
                ],  # last element is the probability
                is_dendrite=dendrite_predictions[idx],
                dendrite_probability=dendrite_probabilities[idx][
                    -1
                ],  # last element is the probability
            )

        roi_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="roi",
        )
        roi_traces_series = RoiResponseSeries(
            name="ROI_fluorescence_timeseries",
            data=roi_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps=plane["timestamps"],
        )

        assert roi_traces.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        neuropil_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="neuropil",
        )
        neuropil_traces_series = RoiResponseSeries(
            name="neuropil_fluorescence_timeseries",
            data=neuropil_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps=plane["timestamps"],
        )

        assert neuropil_traces.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        neuropil_corrected, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="corrected",
        )
        neuropil_corrected_series = RoiResponseSeries(
            name="neuropil_corrected_timeseries",
            data=neuropil_corrected.T,
            rois=roi_names,
            unit="a.u.",
            timestamps=plane["timestamps"],
        )

        assert neuropil_corrected.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        dfof_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["dff_h5"],
            plane_segmentation,
            h5_key="data",
        )

        dfof_traces_series = RoiResponseSeries(
            name="dff_timeseries",
            data=dfof_traces.T,
            rois=roi_names,
            unit="%",
            timestamps=plane["timestamps"],
        )

        assert dfof_traces.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        event_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["events_oasis_h5"],
            plane_segmentation,
            h5_key="events",
        )

        event_traces_series = RoiResponseSeries(
            name="event_timeseries",
            data=event_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps=plane["timestamps"],
        )

        assert event_traces.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        ophys_module.add(
            DfOverF(
                roi_response_series=dfof_traces_series, name="dff_timeseries"
            )
        )

        ophys_module.add(
            Fluorescence(
                roi_response_series=roi_traces_series, name="raw_timeseries"
            )
        )
        ophys_module.add(neuropil_traces_series)
        ophys_module.add(neuropil_corrected_series)
        ophys_module.add(event_traces_series)
    return nwbfile


def find_latest_processed_folder(input_directory: Path) -> Path:
    """
    Find the latest processed asset in the /data directory,
        supporting both singleplane and multiplane.

    Parameters
    ----------
    input_directory : Path
        The directory to search for processed assets.

    Returns
    -------
    Path
        The path to the latest processed asset.
    """
    input_directory = Path(input_directory)

    # Look for folders with "multiplane-ophys" and "processed"
    multiplane_folders = sorted(
        [
            folder
            for folder in input_directory.glob("*")
            if folder.is_dir()
            and "multiplane-ophys" in folder.name
            and "processed" in folder.name
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if multiplane_folders:
        return multiplane_folders[0]

    # Look for a general "processed" folder (singleplane case)
    processed_folders = sorted(
        [
            folder
            for folder in input_directory.glob("*")
            if folder.is_dir() and "processed" in folder.name
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if processed_folders:
        return processed_folders[0]  # Latest processed singleplane folder

    raise FileNotFoundError(
        "No matching processed folder found in the input directory."
    )


def find_latest_raw_folder(input_directory: Path) -> Path:
    """
    Find the latest raw asset in the /data directory,
        supporting both singleplane and multiplane.

    Parameters
    ----------
    input_directory : Path
        The directory to search for raw assets.

    Returns
    -------
    Path
        The path to the latest raw asset.
    """
    input_directory = Path(input_directory)

    # Look for folders with "multiplane-ophys" but NOT "processed"
    multiplane_folders = sorted(
        [
            folder
            for folder in input_directory.glob("*")
            if folder.is_dir()
            and "multiplane-ophys" in folder.name
            and "processed" not in folder.name
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if multiplane_folders:
        return multiplane_folders[0]  # Latest raw multiplane folder

    # Look for general raw folders (singleplane case)
    raw_folders = sorted(
        [
            folder
            for folder in input_directory.glob("*")
            if folder.is_dir() and "raw" in folder.name
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if raw_folders:
        return raw_folders[0]  # Latest raw singleplane folder

    # If no explicit "raw" folder,
    # assume any folder without "processed" is raw
    all_folders = sorted(
        [
            folder
            for folder in input_directory.glob("*")
            if folder.is_dir() and "processed" not in folder.name
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if all_folders:
        return all_folders[0]  # Latest folder assuming it's raw

    raise FileNotFoundError(
        "No matching raw folder found in the input directory."
    )


def set_io_class_backend(
    input_nwb_path: Path, output_nwb: Path
) -> Union[NWBHDF5IO, NWBZarrIO]:
    """Get the IO class based on the file extension

    Parameters
    ----------
    input_nwb_path : Path
        The path to the input NWB file
    output_nwb : Path
        The path to the output NWB file
    Returns
    -------
    NWBHDF5IO
        The IO class
    """
    if input_nwb_path.is_dir():
        assert (
            input_nwb_path / ".zattrs"
        ).is_file(), f"{input_nwb_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = NWBZarrIO
        shutil.copytree(input_nwb_path, output_nwb, dirs_exist_ok=True)
    else:
        NWB_BACKEND = "hdf5"
        io_class = NWBHDF5IO
        shutil.copy(input_nwb_fp, output_nwb_fp)
    logging.info(f"NWB backend: {NWB_BACKEND}")
    return io_class


def sync_times_to_multiplane_fovs(
    ophys_fovs: list, sync_timestamps: np.array
) -> list:
    """Convert the timestamps to FOVs for multiplane only

    Parameters
    ----------
    ophys_fovs : list
        The FOVs
    sync_timestamps : np.array
        The sync timestamps

    Returns
    -------
    list
        The FOVs with timestamps
    """
    planes = len(ophys_fovs)
    if planes % 2 != 0:
        raise Exception("Odd number of planes, please check code")
    # Planes are paired, so we only want to get half of them
    image_groups = int(planes / 2)
    for plane_group in range(image_groups):
        for indiv_plane_in_group in range(2):
            index_plane = plane_group * 2 + indiv_plane_in_group
            # We double check, this is important
            if plane_group != ophys_fovs[index_plane]["coupled_fov_index"]:
                raise Exception(
                    "Mismatched temporal group, please check code"
                )
            ophys_fovs[index_plane]["timestamps"] = sync_timestamps[
                plane_group::image_groups
            ]
    return ophys_fovs


def get_data_paths(input_directory: Path) -> Tuple[Path, Path, Path]:
    """Get the paths to the input NWB file and the processed and raw folders

    Parameters
    ----------
    input_directory : Path
        The directory containing the NWB file and the processed and raw folders

    Returns
    -------
    Tuple[Path, Path, Path]
        The paths to the input NWB file, the processed folder,
        and the raw folder
    """
    input_nwb_paths = list(input_directory.rglob("nwb/*.nwb"))
    if len(input_nwb_paths) != 1:
        raise AssertionError(
            "One valid NWB file must be present in the input directory"
        )
    input_nwb_path = input_nwb_paths[0]
    processed_path = find_latest_processed_folder(args.input_directory)
    raw_path = find_latest_raw_folder(args.input_directory)

    return input_nwb_path, processed_path, raw_path


def get_processed_file_paths(
    processed_path: Path, raw_path: Path, fovs: List, single_plane: bool
) -> dict:
    """Get the paths to the processed files,
       distinguishing between singleplane and multiplane.

    Parameters
    ----------
    processed_path : Path
        The path to the processed folder.
    raw_path : Path
        The path to the raw folder.
    fovs : List
        The list of fovs for processed data.
    single_plane: bool
        Whether the path is single-plane

    Returns
    -------
    dict
        The paths to the processed files.
    """
    file_paths = defaultdict(dict)

    # Determine if it's a singleplane session based on folder name
    is_singleplane = single_plane

    if is_singleplane:
        # Singleplane case: processed_path itself contains the data
        plane_key = processed_path.name  # Use the folder name as the key
        file_paths["planes"][
            plane_key
        ] = file_handling.singleplane_session_data_files(processed_path)
        file_paths["planes"][plane_key][
            "processed_plane_path"
        ] = processed_path
    else:
        # Multiplane case: Extract plane folders
        processed_plane_paths = file_handling.plane_paths_from_session(
            processed_path, data_level="processed", fovs=fovs
        )
        for plane_path in processed_plane_paths:
            file_paths["planes"][plane_path] = file_handling.multiplane_session_data_files(
                processed_path, plane_path
            )
        file_paths["planes"][plane_path]["processed_plane_path"] = plane_path
    file_paths["processed_path"] = processed_path
    file_paths["raw_path"] = raw_path
    return file_paths


def get_metadata(raw_path: Path) -> Tuple[dict, dict, dict]:
    """Get the metadata from the raw folder
    Parameters
    ----------
    raw_path : Path
        The path to the raw folder
    Returns
    -------
    Tuple[dict, dict, dict]
        The session, subject, and rig metadata
    """
    rig_json_path = raw_path / "rig.json"
    if not rig_json_path.is_file():
        raise FileNotFoundError(
            f"Rig JSON file not found in the raw folder, {rig_json_path}"
        )
    session_json_path = raw_path / "session.json"
    if not session_json_path.is_file():
        raise FileNotFoundError(
            f"Session JSON file not found in the raw folder, \
              {session_json_path}"
        )
    subject_json_path = raw_path / "subject.json"
    if not subject_json_path.is_file():
        raise FileNotFoundError(
            f"Subject JSON file not found in the raw folder,\
              {subject_json_path}"
        )

    with open(rig_json_path) as f:
        rig_json_data = json.load(f)
    with open(session_json_path) as f:
        session_json_data = json.load(f)
    with open(subject_json_path) as f:
        subject_json_data = json.load(f)
    return session_json_data, subject_json_data, rig_json_data


def _sync_timestamps(sync_fp: Path) -> np.array:
    """Get the sync timestamps from the sync file
    Parameters
    ----------
    sync_fp : Path
        The path to the sync file
    Returns
    -------
    np.array
        The sync timestamps
    """
    sync_dataset = sync.load_sync(sync_fp)
    print(sync_dataset)
    return sync.get_edges(
        sync_file=sync_dataset,
        kind="rising",
        keys=["vsync_2p", "2p_vsync"],
        units="seconds",
    )


def get_sync_timestamps(raw_path: Path) -> np.array:
    """Get the sync timestamps from the appropriate sync file.

    Parameters
    ----------
    raw_path : Path
        The path to the raw folder.

    Returns
    -------
    np.array
        The sync timestamps.
    """
    # Single-plane check: Look for `data_main.h5`
    sync_fp = next(raw_path.rglob("*pophys/data_main.h5"), None)

    if not sync_fp:
        # Check for the usual multiplane sync files
        sync_fp = next(raw_path.rglob("behavior/*.h5"), None)
    if not sync_fp:
        sync_fp = next(raw_path.rglob("*ophys/*.h5"), None)

    if not sync_fp:
        logging.error(
            "Sync file not found in behavior, *ophys, or single-plane folder"
        )
        raise FileNotFoundError(
            "Sync file not found in behavior, *ophys, or single-plane folder"
        )

    return _sync_timestamps(sync_fp)


def add_intervals_sp_nwb(json_path, frame_rate, nwbfile):
    with open(json_path, "r") as f:
        trial_data = json.load(f)

    # Group TIFFs by their base name
    tiff_groups = {}
    for tiff_name, (start_frame, stop_frame) in trial_data.items():
        # Extract the base name and index from the TIFF filename
        match = re.match(r"([a-zA-Z0-9_]+)_\d+\.tif", tiff_name)
        if match:
            base_name = match.group(1)
            if base_name not in tiff_groups:
                tiff_groups[base_name] = []
            # Convert frames to time
            start_time = start_frame / frame_rate
            stop_time = stop_frame / frame_rate
            tiff_groups[base_name].append((start_time, stop_time))

    # Add intervals for each group of TIFFs
    for base_name, intervals in tiff_groups.items():
        # Create a new TimeIntervals object for this group
        trial_intervals = TimeIntervals(name=base_name)

        # Add the rows for this group of TIFFs
        for start_time, stop_time in intervals:
            trial_intervals.add_row(
                start_time=start_time, stop_time=stop_time
            )

        # Add this TimeIntervals object to the NWB file
        nwbfile.add_time_intervals(trial_intervals)

    return nwbfile


def get_frame_rate(session: dict):
    """Attempt to pull frame rate from session.json
    Returns none if frame rate not in session.json

    Parameters
    ----------
    session: dict
        session metadata

    Returns
    -------
    frame_rate: float
        frame rate in Hz
    """
    frame_rate_hz = None
    for i in session.get("data_streams", ""):
        if i.get("ophys_fovs", ""):
            frame_rate_hz = i["ophys_fovs"][0]["frame_rate"]
            break
    if isinstance(frame_rate_hz, str):
        frame_rate_hz = float(frame_rate_hz)
    return frame_rate_hz


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    Returns
    -------
    argparse.Namespace
        The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert ophys dataset to NWB"
    )

    parser.add_argument(
        "--input_directory",
        type=str,
        help="Path to the input directory",
        default="/data/",
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to the output file",
        default="/results/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)

    input_nwb_fp, processed_data_fp, raw_data_fp = get_data_paths(
        input_directory
    )
    input_nwb_paths = list(input_directory.rglob("nwb/*.nwb"))
    input_nwb_fp = input_nwb_paths[0]

    session_data, subject_data, rig_data = get_metadata(raw_data_fp)
    print(session_data['data_streams'])
    try:
        ophys_fovs = session_data["data_streams"][1]["ophys_fovs"]
    except IndexError:
        ophys_fovs = session_data["data_streams"][0]["ophys_fovs"]        
    single_plane = False
    multiplane = False
    json_path = r"/data/raw/data_description.json"
    with open(json_path, "r") as f:
        data_description = json.load(f)

        if (
            data_description.get("platform", {}).get("abbreviation")
            == "single-plane-ophys"
        ):
            single_plane = True
        else:
            multiplane = True

    file_paths = get_processed_file_paths(
        processed_data_fp, raw_data_fp, ophys_fovs, single_plane
    )
    current_time = datetime.now()
    formatted_date = current_time.strftime("%Y-%m-%d")
    formatted_time = current_time.strftime("%H-%M-%S")
    # determine if file is zarr or hdf5, and copy it to results
    output_nwb_fp = (
        output_directory
        / f"{input_nwb_fp.stem}_processed_ \
        {formatted_date}_{formatted_time}.nwb"
    )
    io_class = set_io_class_backend(input_nwb_fp, output_nwb_fp)
    name_space = "/data/schemas/ndx-aibs-behavior-ophys.namespace.yaml"
    if not Path(name_space).is_file():
        raise FileNotFoundError(name_space)
    # OphysMetadata = load_pynwb_extension("", name_space)
    io = io_class(
        str(output_nwb_fp),
        "r+",
        load_namespaces=True,
        # extensions=name_space,
    )
    nwb_file = io.read()
    if single_plane:
        session_json_path = "/data/raw/session.json"

        with open(session_json_path, "r") as f:
            session_json = json.load(f)
        frame_rate = get_frame_rate(session_json)
        paths = glob.glob(
            "/data/processed/*/motion_correction/trial_locations.json"
        ) + glob.glob("/data/processed/trial_locations.json")

        # Grab the first if any found
        sp_interval_path = paths[0] if paths else None

        nwb_file = add_intervals_sp_nwb(
            sp_interval_path, frame_rate, nwb_file
        )

        nwb_file = nwb_ophys_single_plane(
            nwb_file,
            file_paths,
            rig_data,
            session_data,
            subject_data,
            frame_rate,
        )
        io.write(nwb_file)

    elif multiplane:
        sync_timestamps = get_sync_timestamps(raw_data_fp)
        ophys_fovs = sync_times_to_multiplane_fovs(ophys_fovs, sync_timestamps)
        nwbfile = nwb_ophys(
            nwb_file,
            file_paths,
            ophys_fovs,
            rig_data,
            session_data,
            subject_data,
        )
        # Add plane metadata for each plane
        for fov in ophys_fovs:
            plane_metadata = OphysMetadata(
                name=f'{fov["targeted_structure"]}_{fov["index"]}',
                imaging_depth=str(fov["imaging_depth"]),
                imaging_plane_group=str(fov["coupled_fov_index"]),
                field_of_view_width=str(fov["fov_width"]),
                field_of_view_height=str(fov["fov_height"]),
            )

            # Add the lab_metadata to the NWB file
            nwb_file.add_lab_meta_data(plane_metadata)
        logging.info(nwb_file)

        # write out
        output_directory = Path(args.output_directory).absolute()
        logging.info(f"Writing to {output_directory}")
        io.write(nwb_file)
