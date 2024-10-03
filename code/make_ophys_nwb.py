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

#from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.image import Images, GrayscaleImage
from pynwb.ophys import (
    Fluorescence,
    DfOverF,
    ImageSegmentation,
    MotionCorrection,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)

import file_handling

# local dev: python make_ophys_nwb.py --processed_path ../data/multiplane-ophys_739564_2024-08-26_14-35-58_processed_2024-08-28_21-48-36

CELL_SPECIMEN_COL_DESCRIPTIONS = {
    'cell_specimen_id': 'Unified id of segmented cell across experiments '
                        '(after cell matching)',
    'height': 'Height of ROI in pixels',
    'width': 'Width of ROI in pixels',
    'mask_image_plane': 'Which image plane an ROI resides on. Overlapping '
                        'ROIs are stored on different mask image planes.',
    'max_correction_down': 'Max motion correction in down direction in pixels',
    'max_correction_left': 'Max motion correction in left direction in pixels',
    'max_correction_up': 'Max motion correction in up direction in pixels',
    'max_correction_right': 'Max motion correction in right direction in '
                            'pixels',
    'valid_roi': 'Indicates if cell classification found the ROI to be a cell '
                 'or not',
    'x': 'x position of ROI in Image Plane in pixels (top left corner)',
    'y': 'y position of ROI in Image Plane in pixels (top left corner)'
}

def roi_table_to_pixel_masks_old2(table):
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


def roi_table_to_pixel_masks(table):
    masks = []
    for index, roi in table.iterrows():
        x0 = roi.x
        y0 = roi.y
        mask_matrix = np.array(roi.mask_matrix)

        # Use meshgrid to create coordinate arrays
        y, x = np.meshgrid(np.arange(mask_matrix.shape[1]), np.arange(mask_matrix.shape[0]))

        # Only keep coordinates where mask_matrix is non-zero
        non_zero = mask_matrix != 0
        x = x[non_zero]
        y = y[non_zero]
        z = mask_matrix[non_zero]

        # Add offsets
        x = x + x0
        y = y + y0

        pixel_mask = np.stack((x, y, z), axis=1)
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
    # raw_path = Path(file_paths["raw_path"])
    # session_name = raw_path.name
    processed_session_name = file_paths["processed_path"].name
    session_name = "_".join(file_paths["processed_path"].name.split("_")[:4])

    # get time from session_name
    # dt = "_".join(session_name.split("_")[-2:])
    # converted_dt = datetime.strptime(dt, "%Y-%m-%d_%H-%M-%S").astimezone(tzlocal())

    

    md = file_handling.metadata_for_multiplane(file_paths["processed_path"])

    start_time = md['session_start_time']
    start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ").astimezone(tzlocal())

    # 1. set up the NWB file
    expt_description = \
    f"""
    2-photon calcium imaging in regions {md['session_targeted_structures']} of the mouse brain.
    Recordings were made as the mouse enagages int the following task or views these stimuli.
    {md['session_task_description']}
    """

    nwbfile = NWBFile(
        session_description=md["session_type"],
        identifier=processed_session_name,
        session_start_time=start_time,
        experimenter=md['experimenter'],
        institution="Allen Institute for Neural Dynamics",
        experiment_description=expt_description,
        session_id=session_name,
    )

    # 2. metadata: microscope and optical channels
    device = nwbfile.create_device(
        name=md["microscope_name"],
        description=md["microscope_description"],
    )

    # Could do red/green if we have multiple
    optical_channel = OpticalChannel(
        name="0", # TODO
        description="2P Optical Channel", # TODO
        emission_lambda=510.0, # TODO
    )

    # Start plane specific processing
    for plane_name, plane_files in file_paths["planes"].items():
        plane_path = Path(plane_files["processed_plane_path"])
        plane_name = plane_path.name
        print(f"Adding plane: {plane_name}")

        
        # if error, likely using v4 pipeline outputs, where plane_name/lims_experiment_id mapping
        # is not yet known. raise error. 
        if plane_name not in md['ophys_fovs']:
            raise ValueError(f"Plane name {plane_name} not found in ophys_fovs. \n\
            Check that you are using the correct version of the capsule.")
        else:
            plane_md = md['ophys_fovs'][plane_name]

        plane_description = \
        f"""
        ({plane_md['fov_width']}, {plane_md['fov_height']}) field of view
        in {plane_md['targeted_structure']}
        at depth {plane_md['imaging_depth']} um
        """

        # Grid spacing: physical position of pixels.
        """
        # Grid spacing: physical position of pixels/voxels
        # - Can be ndarray, list, tuple, Dataset, Array, StrDataset, HDMFDataset, or AbstractDataChunkIterator
        # - Specifies space between pixels in (x, y) or voxels in (x, y, z) directions
        # - Uses the specified unit
        # - Assumes imaging plane is a regular grid
        # - See reference_frame to interpret the grid

        Physical location of the first element of the imaging plane (0, 0) for 2-D data or (0, 0, 0) for 3-D data. 
        See also reference_frame for what the physical location is relative to (e.g., bregma).
        """
        imaging_plane = nwbfile.create_imaging_plane(
            name=plane_name,
            optical_channel=optical_channel,
            imaging_rate=float(plane_md['frame_rate']),  # Changed from 'imaging_rate' to 'frame_rate'
            description=plane_description,
            device=device,
            excitation_lambda=float(md['laser_wavelength']),
            indicator=md['gcamp'],
            location=plane_md['targeted_structure'],
            grid_spacing=[float(plane_md['fov_scale_factor']), float(plane_md['fov_scale_factor'])],
            grid_spacing_unit=plane_md['fov_scale_factor_unit'],
            origin_coords=[1.0, 2.0, 3.0],  # TODO
            origin_coords_unit="meters",  # TODO
        )

        # 3. ACQUISITION: raw/mc/decrosstalk movies
        # Skipping for now

        # 4. PROCESSING:
        plane_processing_desc = f"Ophys processing module for {plane_name}"
        ophys_module = nwbfile.create_processing_module(
            name="ophys_plane_" + plane_name, description=plane_processing_desc)

        # 4B. Segmentation 
        # Note: (can hold multiple plane segmentations)
        img_seg = ImageSegmentation()

        # TODO: metadata better for plane segmentation
       
        segmentation_table = pd.DataFrame(load_json(plane_files['segmentation_output_json']))

        # Add this code to set the correct dtype for the 'exclusion_labels' column
        if 'exclusion_labels' in segmentation_table.columns:
            # Convert empty lists to empty numpy arrays with dtype 'U'
            # segmentation_table['exclusion_labels'] = segmentation_table['exclusion_labels'].apply(
            #     lambda x: np.array(x, dtype='U') if x else np.array([], dtype='U')
            # )
            segmentation_table = segmentation_table.drop(columns=['exclusion_labels'])

        #seg_dict = seg_table.to_dict(orient='records')
        #print(seg_table.columns.tolist())
        plane_segmentation = img_seg.create_plane_segmentation(
            name=md["ophys_seg_approach"],
            description=md["ophys_seg_descr"],
            imaging_plane=imaging_plane
        )

        for col_name in segmentation_table.columns:
            # the columns 'roi_mask', 'pixel_mask', and 'voxel_mask' are
            # already defined in the nwb.ophys::PlaneSegmentation Object
            if col_name not in [
                "id",
                "mask_matrix",
                "roi_mask",
                "pixel_mask",
                "voxel_mask",
                "exclusion_labels",
            ]:
                # This builds the columns with name of column and description
                # of column both equal to the column name in the cell_roi_table
                plane_segmentation.add_column(
                    col_name,
                    CELL_SPECIMEN_COL_DESCRIPTIONS.get(
                        col_name, "No Description Available"
                    ),
                )

        # go through each roi and add it to the plan segmentation object
        print(segmentation_table.shape)
        for roi_id, table_row in segmentation_table.iterrows():
            mask = table_row.pop("mask_matrix")
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)

            table_row["id"] = roi_id
            
            y, x = np.where(mask > 0)
            weights = mask[y, x]
            x0 = table_row['x']
            y0 = table_row['y']
            x += x0
            y += y0

            # if x or y more than 512 print
            if np.any(x > 512) or np.any(y > 512):
                print(f"ROI {roi_id} has x or y values greater than 512")

            
            pixel_mask = np.column_stack((x, y, weights))
            plane_segmentation.add_roi(pixel_mask=pixel_mask, **table_row.to_dict())
        ophys_module.add(img_seg)

        # 4C. Summary images
        avg_projection = plt.imread(plane_files['average_projection_png'])
        avg_img = GrayscaleImage(name="average_projection",
                                 data=avg_projection,
                                 resolution=float(plane_md['fov_scale_factor']),
                                 description="Average intensity projection of entire session",)

        max_projection = plt.imread(plane_files['max_projection_png'])
        max_img = GrayscaleImage(name="max_projection",
                                 data=max_projection,
                                 resolution=float(plane_md['fov_scale_factor']), 
                                 description="Max intensity projection of entire session",)

        images = Images(name="summary_images",
                        images=[avg_img, max_img],
                        description="Summary images of the two-photon movie")
        ophys_module.add(images)

        # # 4D. Rois
        # roi_indices = seg_table.id
        # rois_list = roi_table_to_pixel_masks(seg_table)
        # for pixel_mask in rois_list:
        #     ps.add_roi(pixel_mask=pixel_mask)

        # # # 4D. ROI response series
        # # TODO: add more  corrected/events)
        dff_traces, roi_ids_dff = load_dff_h5(plane_files['dff_h5'])
        print(dff_traces.shape)
        roi_table_region = plane_segmentation.create_roi_table_region(
            description="segmented cells labeled by id",
            region=slice(len(dff_traces)))

        # dfof
        roi_response_series = RoiResponseSeries(
            name="traces",
            data=dff_traces.T,
            rois=roi_table_region,
            unit='ΔF/F',
            rate=float(plane_md['frame_rate'])),  # Changed from 'imaging_rate' to 'frame_rate'
            # timestamps =)  #TODO: add timestamps

        dfof = DfOverF(name = "dff", roi_response_series=roi_response_series)
        ophys_module.add(dfof)

    return nwbfile


def attached_dataset():
    processed_path = Path("/root/capsule/data/multiplane-ophys_645814_2022-11-10_15-27-52_processed_2024-02-20_18-31-35")
    raw_path = Path("/root/capsule/data/multiplane-ophys_645814_2022-11-10_15-27-52")
    return processed_path, raw_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ophys dataset to NWB")
    parser.add_argument("--processed_path", type=str, help="Path to the processed ophys session folder", default = r'../data/multiplane-ophys_645814_2022-11-10_15-27-52_processed_2024-02-20_18-31-35')
    parser.add_argument("--raw_path", type=str, help="Path to the raw ophys session folder",default=r'../data/multiplane-ophys_645814_2022-11-10_15-27-52')
    parser.add_argument("--output_path", type=str, help="Path to the output file", default="../results/")
    parser.add_argument("--run_attached", action='store_true')
    args = parser.parse_args()

    if args.run_attached:
        processed_path, raw_path = attached_dataset()
    else:
        processed_path = args.processed_path
        raw_path = args.raw_path

    # file handling & build dict for well known data files
    file_paths = file_handling.get_multiplane_processed_file_paths(processed_path)

    # make NWB
    nwbfile = nwb_ophys(file_paths)

    # write out
    output_path = Path(args.output_path).absolute()
    print(f"Writing to {output_path}")
    with NWBHDF5IO(str(output_path / "ophys.nwb"), "w") as io:
        io.write(nwbfile)
