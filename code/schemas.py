import numpy as np
from marshmallow import RAISE, Schema, fields

STYPE_DICT = {fields.Float: 'float', fields.Int: 'int',
              fields.String: 'text', fields.List: 'text',
              fields.DateTime: 'text', fields.UUID: 'text'}
TYPE_DICT = {fields.Float: float, fields.Int: int, fields.String: str,
             fields.List: np.ndarray, fields.DateTime: str, fields.UUID: str}


class RaisingSchema(Schema):
    class Meta:
        unknown = RAISE


class OphysMetadata(RaisingSchema):
    """This schema contains metadata pertaining to optical physiology (ophys).
    """
    imaging_depth = fields.String(
        doc=('Depth (microns) below the cortical surface '
             'targeted for two-photon acquisition'),
        required=True,
    )
    field_of_view_width = fields.String(
        doc='Width of optical physiology imaging plane in pixels',
        required=True,
    )
    field_of_view_height = fields.String(
        doc='Height of optical physiology imaging plane in pixels',
        required=True,
    )
    imaging_plane_group = fields.String(
        doc=('A numeric index which indicates the order that an imaging plane '
             'was acquired for a mesoscope experiment. Will be -1 for '
             'non-mesoscope data'),
        required=True
    )


