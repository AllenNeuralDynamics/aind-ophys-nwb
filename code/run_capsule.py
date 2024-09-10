from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Union
import json
import h5py

import matplotlib.pyplot as plt
import numpy as np
from dateutil.tz import tzlocal
import pandas as pd
import argparse

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

# capsule
import metadata as md
import file_handling

#from comb.behavior_ophys_dataset import BehaviorOphysDataset

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

def load_dff_h5(dff_file: Union[str,Path],
                remove_nan_rows = False):
    with h5py.File(dff_file, "r") as f:
        dff = f["data"][:]
        roi_names = f["roi_names"][:]
 
    if remove_nan_rows:
        # if row all nans, remove from dff and roi_names
        nan_rows = np.isnan(dff).all(axis=1)
        dff = dff[~nan_rows]
        roi_names = roi_names[~nan_rows]

    return dff, roi_names

def df_col_to_array(df:pd.DataFrame, col:str)->np.ndarray:
    return np.stack(df[col].values)

def nwb_ophys(file_paths: dict):

    # NOTE: could grab= session start time from _json
    raw_path = Path(file_paths["raw_path"])
    session_name = raw_path.name
    dt = "_".join(session_name.split("_")[-2:])
    converted_dt = datetime.strptime(dt, "%Y-%m-%d_%H-%M-%S").astimezone(tzlocal())

    # 1. set up the NWB file
    nwbfile = NWBFile(
        session_description="my first synthetic recording", #TODO
        identifier=str(uuid4()), # TODO
        session_start_time=converted_dt, 
        experimenter=["Baggins, Bilbo"], # TODO
        lab="Bag End Laboratory", # TODO
        institution="Allen Institute for Neural Dynamics",
        experiment_description="I went on an adventure to reclaim vast treasures.", # TODO
        session_id=session_name,
    )


    # 2. metadata: microscope and optical channels
    device = nwbfile.create_device(
        name=md.microscope_name,
        description=md.microscope_desc,
        manufacturer=md.microscope_manufacturer,
    )

    optical_channel = OpticalChannel(
        name=md.oc1_name,
        description=md.oc1_desc,
        emission_lambda=md.oc1_el,
    )

    # Start plane specific processing
    for plane_name, plane_files in file_paths["planes"].items():
        plane_path = Path(plane_files["processed_plane_path"])
        plane_name = plane_path.name
        print(f"Adding plane: {plane_name}")
        # 3. imaging plane 
        imaging_plane = nwbfile.create_imaging_plane(
            name=plane_name, # ophys_plane_id
            optical_channel=optical_channel,
            imaging_rate=1234.0, # TODO
            description="", # TODO
            device=device,
            excitation_lambda=920.0, # TODO: double check, may have changed?
            indicator="GFP", # TODO: subject/procedure.json
            location="VISp", # TODO: session.json
            grid_spacing=[0.01, 0.01], # TODO: dunno
            grid_spacing_unit="meters", # TODO: dunno
            origin_coords=[1.0, 2.0, 3.0], # TODO: dunno
            origin_coords_unit="meters", #TODO
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
        img_seg = ImageSegmentation()

        
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
            name=md.plane_seg_approach, # TODO: name of segmentation
            description=md.plane_seg_descr, # TODO
            imaging_plane=imaging_plane,
            # columns = [seg_table.valid_roi.values],
            # colnames = ["valid_roi"]
        )

        ophys_module.add(img_seg)


        # 4C. summary images
        


        avg_projection= plt.imread(plane_files['average_projection_png'])
        avg_img = GrayscaleImage(name="average_projection",
                             data=avg_projection,
                             resolution=.78, # pixels/cm # TODO change
                             description="Average intensity projection of entire session",)

        max_projection = plt.imread(plane_files['max_projection_png'])
        max_img = GrayscaleImage(name="max_projection",
                             data=max_projection,
                             resolution=.78, # pixels/cm # TODO change
                             description="Max intensity projection of entire session",)

        images = Images(name="summary_images",
                        images=[avg_img, max_img],
                        description="Summary images of the two-photon movie")
        ophys_module.add(images)

        

        # 4D. ROIS
        roi_indices = seg_table.id
        rois_list = roi_table_to_pixel_masks(seg_table)
        for pixel_mask in rois_list:
            ps.add_roi(pixel_mask=pixel_mask)

        # # 4D. ROI response series (possible: raw,np-corrected,dfof,events)
        rt_region = ps.create_roi_table_region(
            region=list(roi_indices), description="all segmented rois"
        )

        # dfof
        dfof_traces, roi_names_dff = load_dff_h5(plane_files['dff_h5'])
        roi_response_series = RoiResponseSeries(
            name="all_rois",
            data=dfof_traces,
            rois=rt_region,
            unit="deltaF/F",
            rate=10.0) # TODO: FRAME RATE
            # timestamps = # TODO) 

        dfof = DfOverF(roi_response_series=roi_response_series)
        ophys_module.add(dfof)

        # dfof - valid only
        # valid_roi_indices = seg_table[seg_table.valid_roi].index.values
        # valid_dfof_traces = dfof_traces[valid_roi_indices]
        # valid_rt_region = ps.create_roi_table_region(
        #     region=list(valid_roi_indices), description="valid rois"
        # )
        # valid_roi_response_series = RoiResponseSeries(
        #     name="dfof_valid_rois",
        #     data=valid_dfof_traces,
        #     rois=valid_rt_region,
        #     unit="deltaF/F",
        #     rate=dataset.metadata["ophys_frame_rate"])
        # valid_dfof = DfOverF(roi_response_series=valid_roi_response_series)
        # ophys_module.add(valid_dfof)

    return nwbfile


def attached_dataset():
    processed_path = Path("/root/capsule/data/multiplane-ophys_645814_2022-11-10_15-27-52_processed_2024-02-20_18-31-35")
    raw_path = Path("/root/capsule/data/multiplane-ophys_645814_2022-11-10_15-27-52")
    return processed_path, raw_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ophys dataset to NWB")
    parser.add_argument("--processed_path", type=str, help="Path to the processed ophys session folder", default = r'/data/multiplane-ophys_645814_2022-11-10_15-27-52_processed_2024-02-20_18-31-35')
    parser.add_argument("--raw_path", type=str, help="Path to the raw ophys session folder",default=r'/data/multiplane-ophys_645814_2022-11-10_15-27-52')
    parser.add_argument("--output_path", type=str, help="Path to the output file", default="/results/")
    parser.add_argument("--run_attached", action='store_true')
    args = parser.parse_args()

    if args.run_attached:
        processed_path, raw_path = attached_dataset()
    else:
        processed_path = args.processed_path
        raw_path = args.raw_path
    
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

    # make NWB
    nwbfile = nwb_ophys(file_paths)

    # write out
    output_path = Path(args.output_path).absolute()
    print(f"Writing to {output_path}")
    with NWBHDF5IO(str(output_path / "ophys.nwb"), "w") as io:
        io.write(nwbfile)