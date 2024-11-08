from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Union
import json
import h5py
import shutil
import glob
import os 
import re
import sparse

import matplotlib.pyplot as plt
import numpy as np
from dateutil.tz import tzlocal
import pandas as pd
import argparse
import pynwb

from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.image import Images, ImageSeries, GrayscaleImage
from pynwb.ophys import (
    CorrectedImageStack,
    Fluorescence,
    DfOverF,
    ImageSegmentation,
    MotionCorrection,
    OnePhotonSeries,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)

from pynwb.file import MultiContainerInterface, NWBContainer, LabMetaData
from pynwb import register_class
from hdmf.utils import docval

# aind
from aind_metadata_mapper.open_ephys.utils import sync_utils as sync

# nwb extension
from schemas import OphysMetadata

# capsule
import file_handling


data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")


def load_pynwb_extension(schema, path):
    neurodata_type = "OphysMetadataSchema"
    pynwb.load_namespaces(path)
    return pynwb.get_class(neurodata_type, 'ndx-aibs-behavior-ophys')

def overall_segmentation_mask(pixel_masks):
    height, width = pixel_masks[0].shape  # Get height and width from the first mask
    full_image_mask = np.zeros((height, width))  # Initialize an empty mask
    
    for mask in pixel_masks:
        full_image_mask = np.logical_or(full_image_mask, mask)  

    
    return full_image_mask.astype(int)

def roi_table_to_pixel_masks_full_image(table, found_metadata):
    img_height = found_metadata['fov_height']
    img_width = found_metadata['fov_width']
    masks = []
    for index, roi in table.iterrows():
        x0 = roi.x
        y0 = roi.y
        if roi.valid_roi == False:
            continue
        full_image_mask = np.zeros((img_height, img_width))
        
        height = roi.height
        width = roi.width
        for i in range(height):
            for j in range(width):
                full_image_mask[y0 + i, x0 + j] = int(roi.mask_matrix[i][j])  

        
        # Directly append the width x height array (full_image_mask)
        masks.append(full_image_mask)
    return masks


def roi_table_to_pixel_masks(table):
    masks = []
    for index, roi in table.iterrows():
        x0 = roi.x
        y0 = roi.y
        mask_matrix = np.array(roi.mask_matrix)
        x, y = np.where(mask_matrix)
        z = mask_matrix[x, y]
        pixel_mask = np.stack((x + x0, y + y0, z), axis=1)

        masks.append(pixel_mask)

    return masks


def roi_table_to_pixel_masks_OLD(table):
    masks = []
    for index, roi in table.iterrows():
        x = roi.x
        y = roi.y
        mask_matrix = roi.mask_matrix

        # convert to X,Y coordinates, starting from x, y, z = value at x,y
        #x_coords, y_coords = np.where(mask_matrix)
        x_coords, y_coords, z = np.where(mask_matrix)
        x_coords = x_coords + x
        y_coords = y_coords + y
        
        # stack the coordinates
        pixel_mask = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=1)
        masks.append(pixel_mask)

    return masks

def load_json(fp):
    with open(fp, 'r') as f:
        j = json.load(f)
    return j

def load_signals(h5_file: Path, ps: ImageSegmentation, h5_group=None, h5_key=None) -> tuple:
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
    rt_region = ps.create_roi_table_region(
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
        Segmentation masks on full image
    """
    print("h5", h5_file)
    with h5py.File(h5_file, "r") as f:
        masks = f[h5_group][h5_key][:]
    
    return masks

def load_sparse_array(h5_file):
    with h5py.File(h5_file) as f:
        data = f["rois"]["data"][:]
        coords = f["rois"]["coords"][:]
        shape = f["rois"]["shape"][:]

    pixelmasks = sparse.COO(coords,data,shape).todense()
    return pixelmasks

def df_col_to_array(df:pd.DataFrame, col:str)->np.ndarray:
    return np.stack(df[col].values)

def nwb_ophys(nwbfile, file_paths: dict, all_planes_session: list, rig_json_data, session_json_data, subject_json_data):

    #session_name = raw_path.name
    #dt = "_".join(session_name.split("_")[-2:])
    #converted_dt = datetime.strptime(dt, "%Y-%m-%d_%H-%M-%S").astimezone(tzlocal())

    raw_path = Path(file_paths["raw_path"])
    session_json_path = raw_path / "session.json"

    # Read the session.json file and extract the session_start_time
    with session_json_path.open("r") as file:
        session_data = json.load(file)
        session_start_time_str = session_data.get("session_start_time")

    # microscope
    microscope_name = session_json_data['rig_id']
    microscope_desc = rig_json_data['rig_id']
    microscope_manufacturer = "Thorlabs" # TODO UPDATE IN rig.json

    # optical channels
    oc1_name = "TODO"
    oc1_desc = "TODO"
    oc1_el = 514.0 # TODO placeholder on Gcamp6 for now emission lambda
    
    # PlaneSegmentation
    plane_seg_approach = "cellpose-mean" # name of individual plane segmentation method
    plane_seg_descr  = "Cellpose with mean intensity projection"

    # 2. metadata: microscope and optical channels
    device = nwbfile.create_device(
        name=microscope_name,
        description=microscope_desc,
        manufacturer=microscope_manufacturer,
    )

    optical_channel = OpticalChannel(
        name=oc1_name,
        description=oc1_desc,
        emission_lambda=oc1_el,
    )

    overall_metadata = {}
    # Start plane specific processing
    processed_targets = set()
    for plane_name, plane_files in file_paths["planes"].items():
        plane_path = Path(plane_files["processed_plane_path"])
        plane_name = plane_path.name
        print(f"Adding plane: {plane_name}")
        # 3. imaging plane 

        # Find matching metdata
        found_metadata = {}
        # Create a set to track combinations of target and index that have been processed

        for index_plane, indiv_plane_metadata in enumerate(all_planes_session):
            target = indiv_plane_metadata['targeted_structure']
            target_index = target + '_' + str(indiv_plane_metadata['index'])

            # Check if this target and index combination has been processed already
            if target_index not in processed_targets:
                # Check if it matches the metadata structure's index
                if target_index == indiv_plane_metadata['targeted_structure'] + "_" + str(indiv_plane_metadata['index']):
                    found_metadata = indiv_plane_metadata
                    processed_targets.add(target_index)  # Mark this combination as processed
                    break
        plane_name = found_metadata['targeted_structure'] + "_" +  str(found_metadata['index'])
        overall_metadata[plane_name] = found_metadata
        location = "Structure: " + found_metadata['targeted_structure'] + " Depth: " +  str(found_metadata['imaging_depth'])
        imaging_plane = nwbfile.create_imaging_plane(
            name=plane_name, # ophys_plane_id
            optical_channel=optical_channel,
            imaging_rate=float(found_metadata['frame_rate']),
            description="Two-photon imaging plane a",
            device=device,
            excitation_lambda=float(session_json_data['data_streams'][0]['light_sources'][0]['wavelength']),
            indicator = subject_json_data['genotype'], 
            location=location,
            grid_spacing=[float(found_metadata['fov_scale_factor']), float(found_metadata['fov_scale_factor'])], 
            grid_spacing_unit=found_metadata['fov_coordinate_unit'],
            origin_coords=[0.0, 0.0, 0.0], # TODO: dunno
            origin_coords_unit= found_metadata['fov_coordinate_unit'], 
        )

        # 3. ACQUISITION: raw/mc/decrosstalk movies
        # two_p_series = TwoPhotonSeries(): 
        # nwbfile.add_acquisition(two_p_series)

        # 4. PROCESSING:
        plane_desc = \
        f"""
        description:
        """
        ophys_module = nwbfile.create_processing_module(
            name="ophys_plane_" + plane_name, description=plane_desc)

        # 4A. Motion Correction: skip for now
        # e.g. TimeSeries() (XY translation)

        # 4B. Segmentation (can hold multiple plane segmentations)
        img_seg = ImageSegmentation(name="image_segmentation")
        
        
        ps = img_seg.create_plane_segmentation(
            name="cell_specimen_table",
            description= plane_seg_approach + plane_seg_descr, # TODO
            imaging_plane=imaging_plane,
            # columns = [seg_table.valid_roi.values],
            # colnames = ["valid_roi"]
        )

        ophys_module.add(img_seg)

        # 4C. summary images
        avg_projection= plt.imread(plane_files['average_projection_png'])
        avg_img = GrayscaleImage(name="average_projection",
                             data=avg_projection,
                             resolution=float(found_metadata['fov_scale_factor']), # pixels/cm
                             description="Average intensity projection of entire session",)

        max_projection = plt.imread(plane_files['max_projection_png'])
        max_img = GrayscaleImage(name="max_projection",
                             data=max_projection,
                             resolution = float(found_metadata['fov_scale_factor']), # pixels/cm 
                             description="Max intensity projection of entire session",)
        
        print(plane_files)
        print(plane_files['extraction_h5'])
        segmetation_mask = load_generic_group(plane_files['extraction_h5'], h5_group="cellpose", h5_key="masks")
        mask_img = GrayscaleImage(name="segmentation_mask_image",
                            data=segmetation_mask,
                            resolution = float(found_metadata['fov_scale_factor']), # pixels/cm 
                            description="Segmentation projection of entire session",)

        images = Images(name="images",
                        images=[avg_img, max_img, mask_img],
                        description="Summary images of the two-photon movie")
        ophys_module.add(images)

        # 4D. ROIS
        
        rois_shape = load_generic_group(plane_files['extraction_h5'], h5_group="rois", h5_key="shape")
        
        for pixel_mask in load_sparse_array(plane_files['extraction_h5']):
            
            ps.add_roi(image_mask=pixel_mask)

        roi_traces, roi_names = load_signals(plane_files['extraction_h5'], ps, h5_group="traces", h5_key="roi")
        roi_traces_series = RoiResponseSeries(
            name="ROI_fluorescence_timeseries",
            data=roi_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps = found_metadata['timestamps']
            )
        
        assert roi_traces.shape[0] == int(rois_shape[0]), "Mismatch in number of ROIs and traces"

        neuropil_traces, roi_names = load_signals(plane_files['extraction_h5'], ps, h5_group="traces", h5_key="neuropil")
        neuropil_traces_series = RoiResponseSeries(
            name="neuropil_fluorescence_timeseries",
            data=neuropil_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps = found_metadata['timestamps']
            )
        
        assert neuropil_traces.shape[0] == int(rois_shape[0]), "Mismatch in number of ROIs and traces"

        neuropil_corrected, roi_names = load_signals(plane_files['extraction_h5'], ps, h5_group="traces", h5_key="corrected")
        neuropil_corrected_series = RoiResponseSeries(
            name="neuropil_corrected_timeseries",
            data=neuropil_corrected.T,
            rois=roi_names,
            unit="a.u.",
            timestamps = found_metadata['timestamps']
            )

        assert neuropil_corrected.shape[0] == int(rois_shape[0]), "Mismatch in number of ROIs and traces"

        dfof_traces, roi_names = load_signals(plane_files['dff_h5'], ps, h5_key = "data")
        
        dfof_traces_series = RoiResponseSeries(
            name="dff_timeseries",
            data=dfof_traces.T,
            rois=roi_names,
            unit="%",
            timestamps = found_metadata['timestamps']
            )
        
        assert dfof_traces.shape[0] == int(rois_shape[0]), "Mismatch in number of ROIs and traces"

        event_traces, roi_names = load_signals(plane_files['events_oasis_h5'], ps, h5_key='events',)

        event_traces_series = RoiResponseSeries(
            name="event_timeseries",
            data=event_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps = found_metadata['timestamps']
            )

        assert event_traces.shape[0] == int(rois_shape[0]), "Mismatch in number of ROIs and traces"

        ophys_module.add(DfOverF(roi_response_series=dfof_traces_series, name="dff"))

        ophys_module.add(Fluorescence(roi_response_series=roi_traces_series))
        ophys_module.add(neuropil_traces_series)
        ophys_module.add(neuropil_corrected_series)
        ophys_module.add(event_traces_series)
    return nwbfile, overall_metadata


def attached_dataset():
    processed_path = Path("/root/capsule/data/multiplane-ophys_645814_2022-11-10_15-27-52_processed_2024-02-20_18-31-35")
    raw_path = Path("/root/capsule/data/multiplane-ophys_645814_2022-11-10_15-27-52")
    return processed_path, raw_path


def find_latest_processed_folder(input_directory: Path) -> Path:
    """
    Find a processed asset in the /data directory
    
    Parameters
    ----------
    input_directory : Path
        The directory to search for processed assets

    Returns
    -------
    Path
        The path to the latest processed asset in /data/
    """
    # Ensure input_directory is a Path object
    input_directory = Path(input_directory)
    print(input_directory)

    # Search for folders that contain "multiplane-ophys" and "processed" in the name
    for folder in input_directory.glob('*'):
        if folder.is_dir() and "multiplane-ophys" in folder.name and "processed" in folder.name:
            return folder
    
    # If no folder matches, look for a general 'raw' folder as fallback
    proc_asset = next(input_directory.glob('processed'), None)
    if proc_asset and proc_asset.is_dir():
        return proc_asset

    raise FileNotFoundError("No matching processed folder found in the input directory.")

# Function to get the latest raw folder
def find_latest_raw_folder(input_directory: Path) -> Path:
    """
    Find a raw asset in the /data directory
    
    Parameters
    ----------
    input_directory : Path
        The directory to search for raw assets
    
    Returns
    -------
    Path
    """
    # Ensure input_directory is a Path object
    input_directory = Path(input_directory)

    # Search for folders that contain "multiplane-ophys" but not "processed" in the name
    for folder in input_directory.glob('*'):
        if folder.is_dir() and "multiplane-ophys" in folder.name and "processed" not in folder.name:
            return folder

    # If no folder matches, look for a general 'raw' folder as fallback
    raw_asset = next(input_directory.glob('raw'), None)
    if raw_asset and raw_asset.is_dir():
        return raw_asset

    raise FileNotFoundError("No matching raw folder found in the input directory.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert ophys dataset to NWB")
    parser.add_argument("--input_directory", type=str, help="Path to the input directory", default="/data/")
    parser.add_argument("--output_directory", type=str, help="Path to the output file", default="/results/")
    parser.add_argument("--run_attached", action='store_true')
    args = parser.parse_args()
    
    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)
    # These parameters are used to adjust for mesoscope processing. Later on we can fetch from the data
    nb_group_planes = 4
    nb_planes_per_group = 2

    input_nwb_paths = list(input_directory.rglob("nwb/*.nwb"))
    if len(input_nwb_paths) != 1:
        raise AssertionError("enter only 1 nwb!")

    input_nwb_path = input_nwb_paths[0]

    processed_path = find_latest_processed_folder(args.input_directory)
    raw_path = find_latest_raw_folder(args.input_directory)
    # file handling & build dict for well known data files

    processed_plane_paths = file_handling.plane_paths_from_session(processed_path, data_level = "processed")
    file_paths = {}
    file_paths['planes'] = {}
    for plane_path in processed_plane_paths:
        plane_path = Path(plane_path)
        plane_name = plane_path.name
        file_paths['planes'][plane_name] = file_handling.multiplane_session_data_files(plane_path)
        file_paths['planes'][plane_name]["processed_plane_path"] = plane_path
    file_paths["processed_path"] = processed_path
    file_paths["raw_path"] = raw_path
    

    # Construct the primary path
    rig_json_path = os.path.join(raw_path, 'rig.json')

    # Fallback path
    fallback_path = '/data/rig/rig.json'

    # Check if the primary path exists, otherwise use the fallback
    if not os.path.exists(rig_json_path):
        rig_json_path = fallback_path

    session_json_path = os.path.join(raw_path, 'session.json')
    try: 
        sync_path = list(Path(raw_path).glob(r'pophys/*.h5'))[0]
    except Exception:
        print(processed_path)
        sync_path = list(Path(processed_path).glob(r'*.h5'))[0]
    subject_json_path = os.path.join(raw_path, 'subject.json')

    with open(rig_json_path, 'r') as file:
        rig_json_data = json.load(file)
 
    with open(session_json_path, 'r') as file:
        session_json_data = json.load(file)

    with open(subject_json_path, 'r') as file:
        subject_json_data = json.load(file)


    all_planes_session = session_json_data['data_streams'][0]['ophys_fovs']
    
    platform_json_path = list(Path(raw_path).glob(r'pophys/*_platform.json'))[0]

    with open(platform_json_path, 'r') as file:
        platform_json_data = json.load(file)

    # We get all timestamps
    sync_dataset = sync.load_sync(sync_path)
    timestamps = sync.get_edges(sync_file=sync_dataset, kind="rising", keys=["vsync_2p", "2p_vsync"], units="seconds")

    nb_group_planes = 4
    nb_planes_per_group = 2

    for plane_group in range(nb_group_planes):
        for indiv_plane_in_group in range(nb_planes_per_group):
            index_plane = plane_group*nb_planes_per_group+indiv_plane_in_group

            # We double check, this is important
            if plane_group !=  all_planes_session[index_plane]['coupled_fov_index']:
                raise Exception("Mismatched temporal group, please check code")

            all_planes_session[index_plane]['timestamps'] = timestamps[plane_group::nb_group_planes]

    # determine if file is zarr or hdf5, and copy it to results
    result_nwb_path = output_directory / input_nwb_path.name
    if input_nwb_path.is_dir():
        assert (input_nwb_path / ".zattrs").is_file(), f"{input_nwb_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = NWBZarrIO
        shutil.copytree(input_nwb_path, result_nwb_path, dirs_exist_ok=True)
    else:
        NWB_BACKEND = "hdf5"
        io_class = NWBHDF5IO
        shutil.copyfile(input_nwb_path, result_nwb_path)
    print(f"NWB backend: {NWB_BACKEND}")
    OphysMetadata = load_pynwb_extension("", r'/data/schemas/ndx-aibs-behavior-ophys.namespace.yaml')    
    io = io_class(str(result_nwb_path), "r+", load_namespaces=False, extensions=r'/data/schemas/ndx-aibs-behavior-ophys.namespace.yaml')
    nwb_file = io.read()
    nwbfile, overall_metadata = nwb_ophys(nwb_file, file_paths, all_planes_session, rig_json_data, session_json_data, subject_json_data)

    depth = ""
    plane_group = ""
    field_of_view_width = ""
    field_of_view_height = ""

    # Add plane metadata for each plane
    for plane_name, found_metadata in overall_metadata.items():
        plane_metadata = OphysMetadata(
            name=f"{plane_name}",
            imaging_depth=str(found_metadata['imaging_depth']),
            imaging_plane_group=str(found_metadata['coupled_fov_index']),
            field_of_view_width=str(found_metadata['fov_width']),
            field_of_view_height=str(found_metadata['fov_height'])
        )
        
        # Add the lab_metadata to the NWB file
        nwb_file.add_lab_meta_data(plane_metadata)
    print(nwb_file)

    # write out
    output_directory = Path(args.output_directory).absolute()
    print(f"Writing to {output_directory}")
    io.write(nwb_file)