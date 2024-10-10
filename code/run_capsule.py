from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Union
import json
import h5py
import shutil
import glob
import os 

import matplotlib.pyplot as plt
import numpy as np
from dateutil.tz import tzlocal
import pandas as pd
import argparse

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

# aind
from aind_metadata_mapper.open_ephys.utils import sync_utils as sync

# capsule
import file_handling


data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")


def roi_table_to_pixel_masks_full_image(table, found_metadata):
    height = found_metadata['fov_height']
    width = found_metadata['fov_width']
    masks = []
    for index, roi in table.iterrows():
        x0 = roi.x
        y0 = roi.y
        full_image_mask = np.zeros((height, width))
        
        full_image_mask[x0, y0] = 1
        
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

def load_traces_h5(h5_file, ps,  h5_key='data', roi_key = 'roi_names', mask_ids = []): 
    with h5py.File(h5_file, "r") as f:
        traces = f[h5_key][:]
        roi_names =  f[roi_key][:]
        
        final_int_list = []
        
        # We extract the binary list of trace_id in masks
        int_list = [int(x.decode()) for x in roi_names]
        
        # This is the list of ROI that are packaged
        mask_ids_list = list(mask_ids)
        
        # Convert mask_ids_list to a numpy array
        mask_ids_array = np.array(mask_ids_list)

        # We look for the indice of the mask into the list of packaged masks
        # This code should crash if it can't find the ROI indice. This is intentional. 
        indices_in_original_table = [np.where(x==mask_ids_array)[0][0] for i, x in enumerate(int_list)]

        # We only save the indice of the ROI mask that correspond to the trace. 
        rt_region = ps.create_roi_table_region(
            region=indices_in_original_table, description="List of measured ROIs"
        )

    return traces, rt_region

def df_col_to_array(df:pd.DataFrame, col:str)->np.ndarray:
    return np.stack(df[col].values)

def nwb_ophys(nwbfile, file_paths: dict, all_planes_session: list, rig_json_data, session_json_data, subject_json_data):

    # NOTE: could grab= session start time from _json
    raw_path = Path(file_paths["raw_path"])
    session_name = raw_path.name
    dt = "_".join(session_name.split("_")[-2:])
    converted_dt = datetime.strptime(dt, "%Y-%m-%d_%H-%M-%S").astimezone(tzlocal())

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

    # Start plane specific processing
    for plane_name, plane_files in file_paths["planes"].items():
        plane_path = Path(plane_files["processed_plane_path"])
        plane_name = plane_path.name
        print(f"Adding plane: {plane_name}")
        # 3. imaging plane 

        # Find matching metdata
        found_metadata = {}
        for index_plane, indiv_plane_metadata in enumerate(all_planes_session):
            target = indiv_plane_metadata['targeted_structure']
            target+'_'+str(index_plane)
            if target+'_'+str(index_plane) == plane_name:
                found_metadata = indiv_plane_metadata
                break
        
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

        
        #img_stack = average_projection[np.newaxis, :, :]
        # avg_projection = ImageSeries(
        #     name="average_intensity_projection",
        #     data=img_stack,
        #     format="raw",
        #     unit="float32",
        #     rate = 0.0,
        # )

        

        seg_table = pd.DataFrame(load_json(plane_files['segmentation_output_json']))
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

        roi_indices = seg_table.id
        rois_list = roi_table_to_pixel_masks_full_image(seg_table, found_metadata)
        for pixel_mask in rois_list:
            ps.add_roi(image_mask=pixel_mask)

        images = Images(name="images",
                        images=[avg_img, max_img, pixel_mask],
                        description="Summary images of the two-photon movie")
        ophys_module.add(images)

        # 4D. ROIS
        dfof_traces, roi_names = load_traces_h5(plane_files['dff_h5'], ps, mask_ids = roi_indices)
        
        dfof_traces_series = RoiResponseSeries(
            name="deltaFoverF",
            data=dfof_traces.T,
            rois=roi_names,
            unit="%",
            timestamps = found_metadata['timestamps']
            )

        roi_traces, roi_names = load_traces_h5(plane_files['roi_traces_h5'], ps, mask_ids = roi_indices)
        roi_traces_series = RoiResponseSeries(
            name="ROI fluorescence",
            data=roi_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps = found_metadata['timestamps']
            )

        neuropil_traces, roi_names = load_traces_h5(plane_files['neuropil_traces_h5'], ps, mask_ids = roi_indices)
        neuropil_traces_series = RoiResponseSeries(
            name="neuropil_fluorescence",
            data=neuropil_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps = found_metadata['timestamps']
            )

        neuropil_corrected, roi_names = load_traces_h5(plane_files['neuropil_correction_h5'], ps, h5_key='FC', mask_ids = roi_indices)

        neuropil_corrected_series = RoiResponseSeries(
            name="neuropil_corrected",
            data=neuropil_corrected.T,
            rois=roi_names,
            unit="a.u.",
            timestamps = found_metadata['timestamps']
            )

        event_traces, roi_names = load_traces_h5(plane_files['events_oasis_h5'], ps, h5_key='events', roi_key='cell_roi_id', mask_ids = roi_indices)

        event_traces_series = RoiResponseSeries(
            name="Events",
            data=event_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps = found_metadata['timestamps']
            )

        ophys_module.add(DfOverF(roi_response_series=dfof_traces_series, name="dff"))

        ophys_module.add(Fluorescence(roi_response_series=roi_traces_series))
        ophys_module.add(neuropil_traces_series)
        ophys_module.add(neuropil_corrected_series)
        ophys_module.add(event_traces_series)

    return nwbfile


def attached_dataset():
    processed_path = Path("/root/capsule/data/multiplane-ophys_645814_2022-11-10_15-27-52_processed_2024-02-20_18-31-35")
    raw_path = Path("/root/capsule/data/multiplane-ophys_645814_2022-11-10_15-27-52")
    return processed_path, raw_path


def find_latest_processed_folder():
    # Define a glob pattern to match processed folders in /data/
    pattern = '/data/*multiplane-ophys*_processed_*'
    processed_folders = glob.glob(pattern)

    if processed_folders:
        # Sort by modification time and return the latest one
        return max(processed_folders, key=os.path.getmtime)
    else:
        raise FileNotFoundError("No processed folder found in /data/")

# Function to get the latest raw folder
def find_latest_raw_folder():
    # Define a glob pattern to match raw folders in /data/
    pattern = '/data/*multiplane-ophys*'
    raw_folders = [folder for folder in glob.glob(pattern) if 'processed' not in folder]

    if raw_folders:
        # Sort by modification time and return the latest one
        return max(raw_folders, key=os.path.getmtime)
    else:
        raise FileNotFoundError("No raw folder found in /data/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ophys dataset to NWB")
    parser.add_argument("--processed_path", type=str, help="Path to the processed ophys session folder", default = r'/data/multiplane-ophys_731327_2024-08-22_14-11-29_processed_2024-09-25_16-28-44')
    parser.add_argument("--raw_path", type=str, help="Path to the raw ophys session folder",default=r'/data/multiplane-ophys_731327_2024-08-22_14-11-29')
    parser.add_argument("--output_path", type=str, help="Path to the output file", default="/results/")
    parser.add_argument("--run_attached", action='store_true')
    args = parser.parse_args()

    # These parameters are used to adjust for mesoscope processing. Later on we can fetch from the data
    nb_group_planes = 4
    nb_planes_per_group = 2

    input_nwb_paths = list(data_folder.glob(r'nwb/*.nwb'))
    if len(input_nwb_paths) != 1:
        print("enter only 1 nwb!")

    input_nwb_path = input_nwb_paths[0]

    processed_path = find_latest_processed_folder()
    raw_path = find_latest_raw_folder()
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
    
    rig_json_path = os.path.join(raw_path, 'rig.json')
    session_json_path = os.path.join(raw_path, 'session.json')
    sync_path = list(Path(raw_path).glob(r'pophys/*_sync.h5'))[0]
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
    result_nwb_path = results_folder / input_nwb_path.name
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
    # make NWB
    io = io_class(str(result_nwb_path), "r+", load_namespaces=True)
    nwb_file = io.read()
    nwbfile = nwb_ophys(nwb_file, file_paths, all_planes_session, rig_json_data, session_json_data, subject_json_data)

    # write out
    output_path = Path(args.output_path).absolute()
    print(f"Writing to {output_path}")
    io.write(nwb_file)