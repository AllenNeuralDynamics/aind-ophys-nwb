## copied from comb

from pathlib import Path
import json
from typing import Union, Dict, List
import warnings

# set up logger
import logging
logger = logging.getLogger(__name__)



# MULTIPLANE_FILE_PARTS = {"processing_json": "processing.json",
#                            "params_json": "_params.json",
#                            "registered_metrics_json": "_registered_metrics.json",
#                            "average_projection_png": "_average_projection.png",
#                            "max_projection_png": "_maximum_projection.png",
#                            "motion_transform_csv": "_motion_transform.csv",
#                            "segmentation_output_json": "segmentation_output.json",
#                            "roi_traces_h5": "roi_traces.h5",
#                            "neuropil_traces_h5": "neuropil_traces.h5",
#                            "neuropil_correction_h5": "neuropil_correction.h5",
#                            "neuropil_masks_json": "neuropil_masks.json",
#                            "neuropil_trace_output_json": "neuropil_trace_output.json",
#                            #"demixing_h5": "demixing_output.h5",
#                            #"demixing_json": "demixing_output.json",
#                            "dff_h5": "dff.h5",
#                            "extract_traces_json": "extract_traces.json",
#                            "events_oasis_h5": "events_oasis.h5"}

MULTIPLANE_FILE_PARTS = {"processing_json": "processing.json",
                           "params_json": "params.json",
                           "registered_metrics_json": "registered_metrics.json",
                           "average_projection_png": "average_projection.png",
                           "max_projection_png": "maximum_projection.png",
                           "motion_transform_csv": "motion_transform.csv",
                           "extraction_h5": "extraction.h5",
                           "dff_h5": "dff.h5",
                           "extract_traces_json": "extract_traces.json",
                           "events_oasis_h5": "events_oasis.h5"}

def multiplane_session_data_files(input_path, plane):
    """Find all data files in a multiplane session directory."""
    input_path = Path(input_path)
    data_files = {}
    for key, value in MULTIPLANE_FILE_PARTS.items():
        data_files[key] = find_data_file(input_path, plane + "_" + value)
    return data_files


def find_data_file(input_path, file_part, verbose=False):
    """Find a file in a directory given a partial file name.

    Example
    -------
    input_path = /root/capsule/data/multiplane-ophys_724567_2024-05-20_12-00-21
    file_part = "_sync.h5"
    return: "/root/capsule/data/multiplane-ophys_724567_2024-05-20_12-00-21/ophys/1367710111_sync.h5"
    
    
    Parameters
    ----------
    input_path : str or Path
        The path to the directory to search.
    file_part : str
        The partial file name to search for.
    """
    input_path = Path(input_path)
    try:
        file = list(input_path.glob(f'**/*{file_part}*'))[0]
    except IndexError:
        if verbose:
            logger.warning(f"File with '{file_part}' not found in {input_path}")
        file = None
    return file


def get_file_paths_dict(file_parts_dict, input_path):
    file_paths = {}
    for key, value in file_parts_dict.items():
        file_paths[key] = find_data_file(input_path, value)
    return file_paths


def check_ophys_folder(path):
    """ophys folders can have multiple names, check for all of them"""
    ophys_names = ['ophys', 'pophys', 'mpophys']
    ophys_folder = None
    for ophys_name in ophys_names:
        ophys_folder = path / ophys_name
        if ophys_folder.exists():
            break
        else:
            ophys_folder = None

    return ophys_folder


def check_behavior_folder(path):
    behavior_names = ['behavior', 'behavior_videos']
    behavior_folder = None
    for behavior_name in behavior_names:
        behavior_folder = path / behavior_name
        if behavior_folder.exists():
            break
        else:
            behavior_folder = None
    return behavior_folder


def get_sync_file_path(input_path, verbose=False):
    """Find the Sync file"""
    file_parts = {}
    input_path = Path(input_path)
    try: 
        # method 1: find sync_file by name
        file_parts = {"sync_h5": "_sync.h5"}
        sync_file_path = find_data_file(input_path, file_parts["sync_h5"], verbose=False)
    except IndexError as e:
        if verbose:
            logger.info("file with '*_sync.h5' not found, trying platform json")

    if sync_file_path is None:
        # method 2: load platform json
        # Note: sometimes fails if platform json has incorrect sync_file path
        logging.info(f"Trying to find sync file using platform json for {input_path}")
        file_parts = {"platform_json": "_platform.json"}
        platform_path = find_data_file(input_path, file_parts["platform_json"])
        with open(platform_path, 'r') as f:
            platform_json = json.load(f)
        ophys_folder = check_ophys_folder(input_path)
        sync_file_path = ophys_folder / platform_json['sync_file']

        if not sync_file_path.exists():
            logger.error(f"Unsupported data asset structure, sync file not found in {sync_file_path}")
            sync_file_path = None
        else:
            logger.info(f"Sync file found in {sync_file_path}")

    return sync_file_path


def plane_paths_from_session(session_path: Union[Path, str],
                             data_level: str = "raw", fovs: List = []) -> list:
    """Get plane paths from a session directory

    Parameters
    ----------
    session_path : Union[Path, str]
        Path to the session directory
    data_level : str, optional
        Data level, by default "raw". Options: "raw", "processed"
    fovs: List, optional
        List of ophys fovs. Should be provided for all processed assets

    Returns
    -------
    list
        List of plane paths
    """
    if isinstance(session_path, str):
        session_path = Path(session_path)
    if fovs != [] and data_level == "processed":
        fov_pairs = []
        for fov in fovs:
            fov_plane = fov['targeted_structure']
            fov_index = fov['index']
            fov_pair= fov_plane +"_"+ str(fov_index)
            fov_pairs.append(fov_pair)
            planes = [x for x in fov_pairs]
    elif fovs == [] and data_level == "processed":
        logger.error("Processed data requires ophys fovs")
    if data_level == "raw":
        planes = [f for f in (session_path / 'pophys').glob(*) if f.is_dir()]
    return planes


def get_multiplane_processed_file_paths(processed_path: Union[Path, str]) -> dict:
    """
    Create a dictionary of file paths for a processed ophys session.

    Parameters
    ----------
    processed_path : Union[Path, str]
        Path to the processed ophys session folder.

    Returns
    -------
    dict
        A dictionary containing file paths for each plane and the processed path.
    """
    processed_path = Path(processed_path)
    processed_plane_paths = plane_paths_from_session(processed_path, data_level="processed")

    file_paths = {'planes': {}, 'processed_path': processed_path}

    for plane_path in processed_plane_paths:
        plane_path = Path(plane_path)
        plane = plane_path.name
        file_paths['planes'][plane] = multiplane_session_data_files(plane_path)
        file_paths['planes'][plane]["processed_plane_path"] = plane_path

    return file_paths

####################################################################################################
# METADATA EXTRACTION
####################################################################################################


def extract_laser_metadata(session_json: dict):
    """Get laser metadata from the session json.

    Parameters
    ----------
    session_json: dict
        session json

    Returns
    -------
    laser_metadata: dict
    """
    ophys_stream = extract_ophys_stream(session_json)
    light_sources = ophys_stream.get('light_sources', [])
    laser_stream = next((ls for ls in light_sources if ls.get('name') == 'Laser'), None)
    if laser_stream is None:
        raise ValueError("No Laser device found in data streams")
    return {
        'laser_wavelength': laser_stream.get('wavelength'),
        'wavelength_unit': laser_stream.get('wavelength_unit')
    }


def extract_ophys_stream(session_json: dict):
    """
    Extract the ophys stream from the session metadata.

    Parameters
    ----------
    metadata : dict
        The session metadata dictionary.

    Returns
    -------
    dict
        The ophys stream data.

    Raises
    ------
    ValueError
        If no ophys stream is found in the metadata.
    """
    ophys_stream = next((stream for stream in session_json['data_streams'] 
                         if any(modality.get('name') == 'Planar optical physiology' 
                                for modality in stream.get('stream_modalities', []))), 
                        None)

    if not ophys_stream:
        raise ValueError("No ophys stream found in the metadata")
    
    return ophys_stream


def extract_ophys_fovs(session_json: dict):
    """Get a dict of ophys fovs with the index and targeted structure as the key.

    Parameters
    ----------
    session_json: dict
        session json

    Returns
    -------
    structured_fovs: dict
    """
    ophys_stream = extract_ophys_stream(session_json)
    ophys_fovs = ophys_stream.get('ophys_fovs', [])
    structured_fovs = {}

    for fov in ophys_fovs:
        index = fov.get('index')
        targeted_structure = fov.get('targeted_structure')

        if index is not None and targeted_structure is not None:
            key = f"{targeted_structure}_{index}"
            structured_fovs[key] = fov

    return structured_fovs


def gcamp_from_genotype(genotype: str):
    """Get the gcamp version from the genotype.

    Parameters
    ----------
    genotype: str
        The genotype of the subject.

    Returns
    -------
    gcamp_version: str
    """

    genotype_parts = genotype.split('-')
    gcamp_part = next((part for part in genotype_parts if 'gcamp' in part.lower()), None)

    # Check for wildtype genotype
    if genotype.lower() in ['wt/wt', 'wt/wt ']:
        warnings.warn("Wildtype genotype detected. "
                      "Viral injection for GCaMP not implemented.",
                      UserWarning)
        return None

    if gcamp_part:
        return gcamp_part
    else:
        return None  # or you could return a default value or raise an exception


def metadata_for_multiplane(data_path: Union[str, Path]) -> dict:
    """Extract metadata to build nwb ophys file

    Parameters
    ----------
    data_path: Union[str, Path]
        path to the processed/raw multiplane ophys session folder

    Returns
    -------
    metadata: dict

    """

    md = {}

    jsons = load_metadata_json_files(data_path)

    ### SESSION METADATA ###
    md['session_start_time'] = jsons['session'].get('session_start_time')
    md['experimenter'] = jsons['session'].get('experimenter_full_name')
    md['session_type'] = jsons['session'].get('session_type')
    md.update(extract_laser_metadata(jsons['session']))
    md['ophys_fovs'] = extract_ophys_fovs(jsons['session'])

    md['session_targeted_structures'] = list(set(
        fov['targeted_structure'] 
        for fov in md['ophys_fovs'].values()
        if 'targeted_structure' in fov
    ))

    ### SESSION SUMMARY METADATA ###
    md['session_num_planes'] = len(md['ophys_fovs'].keys())

    imaging_depths = []
    for fov in md['ophys_fovs'].values():
        if 'imaging_depth' in fov:
            imaging_depths.append(fov['imaging_depth'])

    md['session_imaging_depths'] = sorted(imaging_depths)

    # This should be a natural text summary of the task
    md['session_task_description'] = "Visual change detection task."

    ### RIG METADATA ###
    # could get from rig, but also in session.json
    md['microscope_name'] = jsons['session'].get('rig_id')
    if md['microscope_name'] in ["MESO.1", "MESO.2"]:
        md['microscope_description'] = "AIND Multiplane Mesoscope 2P Rig"
    else:
        md['microscope_description'] = "Unknown"

    ### SUBJECT METADATA ###
    md['subject_id'] = jsons['subject'].get('subject_id')
    md['genotype'] = jsons['subject'].get('genotype')
    md['sex'] = jsons['subject'].get('sex')
    md['gcamp'] = gcamp_from_genotype(md['genotype'])

    # TODO: better metadata for plane segmentation
    md['ophys_seg_approach'] = "Cellpose"
    md['ophys_seg_descr'] = "Cellpose segmentation of two-photon movie"

    return md


def load_metadata_json_files(processed_path: Union[str, Path]) -> Dict[str, Union[dict, None]]:
    """
    Load procedures.json, session.json, subject.json, and rig.json into a dictionary.

    Parameters
    ----------
    processed_path : Union[str, Path]
        Path to the processed ophys session folder.

    Returns
    -------
    Dict[str, Union[dict, None]]
        A dictionary containing the loaded JSON data for each file.
        If a file is not found, its value will be None.
    """
    processed_path = Path(processed_path)
    json_files = ['procedures.json', 'session.json', 'subject.json', 'rig.json']
    result = {}

    for file_name in json_files:
        file_path = processed_path / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    result[file_name.replace('.json', '')] = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {file_name}. File may be empty or contain invalid JSON.")
                result[file_name.replace('.json', '')] = {}
        else:
            print(f"File {file_name} not found in {processed_path}")
            result[file_name.replace('.json', '')] = {}

    return result
