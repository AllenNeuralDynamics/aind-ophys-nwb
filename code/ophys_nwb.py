from pydantic import BaseModel, Field
from typing import List, Optional
from hdmf_zarr import NWBZarrIO
import pynwb
from pynwb import NWBHDF5IO
from hdmf_zarr import NWBZarrIO
from collections import defaultdict
from pynwb.image import Images, GrayscaleImage
from pynwb.ophys import (
    DfOverF,
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
    RoiResponseSeries,
)


class NWBSettings(BaseModel):
    nwbfile: pynwb.NWBFile = Field(..., description="NWBFile object")
    plane_name: str = Field(..., description="Plane name")
    rig_json_data: dict = Field(..., description="Rig JSON data")
    session_json_data: dict = Field(..., description="Session JSON data")
    subject_json_data: dict = Field(..., description="Subject JSON data")


class OphysNWB:
    def __init__(self, nwb_settings: NWBSettings):
        self.nwb_file = nwb_settings.nwbfile
        self.plane_name = nwb_settings.plane_name
        self.rig_json_data = nwb_settings.rig_json_data
        self.session_json_data = nwb_settings.session_json_data
        self.subject_json_data = nwb_settings.subject_json_data

    def create_microscope(self):
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
        microscope_name = self.session_json_data["rig_id"]
        microscope_desc = self.rig_json_data["rig_id"]
        microscope_manufacturer = "Thorlabs"  # TODO UPDATE IN rig.json
        # optical channels
        oc1_name = "TODO"
        oc1_desc = "TODO"
        oc1_el = 514.0  # TODO placeholder on Gcamp6 for now emission lambda
        device = self.nwbfile.create_device(
            name=microscope_name,
            description=microscope_desc,
            manufacturer=microscope_manufacturer,
        )
        optical_channel = OpticalChannel(
            name=oc1_name,
            description=oc1_desc,
            emission_lambda=oc1_el,
        )
        return device, optical_channel

    def create_plane_segmentation(
        self, seg_approach: str, seg_descr: str, imaging_plane: pynwb.ophys.ImagingPlane
    ) -> Tuple[pynwb.ophys.ImageSegmentation, pynwb.ophys.PlaneSegmentation]:
        """Create a plane segmentation for the NWB file

        Parameters
        ----------
        seg_approach : str
            The segmentation approach
        seg_descr : str
            The segmentation description
        imaging_plane : pynwb.ophys.ImagingPlane
            The imaging plane

        Returns
        -------
        Tuple[pynwb.ophys.ImageSegmentation, pynwb.ophys.PlaneSegmentation]
            The image segmentation and plane segmentation
        """
        img_seg = ImageSegmentation(name="image_segmentation")
        plane_segmentation = img_seg.create_plane_segmentation(
            name="cell_specimen_table",
            description=seg_approach + seg_descr,
            imaging_plane=imaging_plane,
        )
        return img_seg, plane_segmentation

    def create_imaging_plane(
        self,
        fov: dict,
        location: set,
        session_json_data: dict,
        optical_channel: OpticalChannel,
        device: pynwb.device,
    ) -> pynwb.ophys.ImagingPlane:
        """Create an imaging plane for the NWB file

            Parameters
            ----------
            nwbfile : pynwb.NWBFile
                The NWB file
            fov : dict
                The FOV metadata
            location : set
                The location of the imaging plane
            session_json_data : dict
                The session metadata
            optical_channel : OpticalChannel
                The optical channel
            device : pynwb.device
                NWB device

            Returns
            -------
            pynwb.ophys.ImagingPlane
                The imaging plane
            """
            return self.nwbfile.create_imaging_plane(
                name=plane_name,  # ophys_plane_id
                optical_channel=optical_channel,
                imaging_rate=float(fov["frame_rate"]),
                description="Two-photon imaging plane a",
                device=device,
                excitation_lambda=float(
                    session_json_data["data_streams"][0]["light_sources"][0]["wavelength"]
                ),
                indicator=subject_json_data["genotype"],
                location=location,
                grid_spacing=[
                    float(fov["fov_scale_factor"]),
                    float(fov["fov_scale_factor"]),
                ],
                grid_spacing_unit=fov["fov_coordinate_unit"],
                origin_coords=[0.0, 0.0, 0.0],  # TODO: dunno
                origin_coords_unit=fov["fov_coordinate_unit"],
            )

    def create_ophys_module(
        self, plane_name: str, plane_meta: dict, location: set, optical_channel, device
    ) -> Tuple[pynwb.NWBFile]:
        """Create the ophys module for an individual plane

        Parameters
        ----------
        plane_name : str
            The name of the plane
        plane_meta : dict
            The metadata for the plane
        location : set
            The location of the plane
        optical_channel : OpticalChannel
            The optical channel
        device : Device
            The device

        Returns
        -------
        NWBFile
            The NWB file
        """
        ophys_module = self.nwbfile.create_processing_module(name=plane_name, description="")
        imaging_plane = self.create_imaging_plane(
            self.nwbfile,
            plane_meta,
            location,
            self.session_json_data,
            optical_channel,
            device,
        )
        return ophys_module, imaging_plane
