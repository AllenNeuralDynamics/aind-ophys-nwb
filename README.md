# AIND Ophys NWB

Packages the outputs of the AIND ophys processing pipeline into a single NWB
(Zarr) asset, for single-plane and multiplane optical physiology datasets.

This capsule is a **thin wrapper**: all logic lives in the
[`aind-ophys-nwb-library`](https://github.com/AllenNeuralDynamics/aind-ophys-nwb-library)
package, and `code/run_capsule.py` only parses settings and calls
`aind_ophys_nwb_library.job.run`.

## Position in the pipeline

NWB packaging is the pipeline's **terminal fan-in**: `main.nf` invokes it once
with every upstream capsule's results `.collect()`ed, so a single run sees all
planes at once. It reads what motion correction, decrosstalk, extraction,
dF/F, event detection and ROI classification each wrote.

## Input

Parameters are a `pydantic-settings` model (`NwbSettings` in the library) and
are passed as `python run_capsule.py --name=value`, which is the only form a
Code Ocean app panel emits. Hyphenated flags are **not** accepted.

| parameter | default | meaning |
|---|---|---|
| `--input_dir` | `/data` | mounted inputs |
| `--output_dir` | `/results` | where the NWB asset is written |
| `--raw_subdir` | `raw` | subdirectory of `input_dir` holding the raw asset |
| `--processed_subdir` | `processed` | subdirectory holding the processed results |
| `--nwb_name` | `pophys.nwb.zarr` | basename of the NWB Zarr store. Written into a subdirectory of `output_dir`, never at the results root, so the pipeline's basename-flattening `publishDir` cannot collide |
| `--extension_namespace` | `ndx-aibs-behavior-ophys.namespace.yaml` | the pynwb extension namespace; `code/run` passes an absolute path anchored on `code/` |
| `--skip_lab_metadata` | `false` | skip the per-plane `ndx-aibs-behavior-ophys` lab metadata |
| `--verify` | `false` | round-trip the emitted `processing.json` and `quality_control.json` back through the core aind-data-schema v2 objects |

### Runtime assets

`ndx-aibs-behavior-ophys.namespace.yaml` and
`ndx-aibs-behavior-ophys.extension.yaml` define the custom `OphysMetadataSchema`
lab metadata and are committed under `code/`. That location is required:
`main.nf` stages **only** a capsule's `code/` directory, so a file elsewhere in
the repo is present in a capsule run but absent in every pipeline run.
`code/run` checks both exist before any work starts and passes the namespace
path explicitly, so resolution does not depend on the working directory.

## Output

- the NWB Zarr store, under a subdirectory of `output_dir`
- a full `processing.json` — an aind-data-schema v2 `Processing`
- a full `quality_control.json` — a v2 `QualityControl` with
  `default_grouping = ["evaluation"]`

Metadata is emitted against **aind-data-schema 2.x**. `QCEvaluation` does not
exist in v2, so metric grouping is carried by each metric's `evaluation` tag
rather than by a wrapping evaluation object.

## Examining NWB content

`code/` ships notebooks for loading and inspecting the result:
`load_ophys_only_nwb.ipynb`, `load_ophys_only_nwb_correct_specimen_info.ipynb`
and `nwb_visualization_notebook.ipynb`.

The NWB contains:

1. a processing container with the per-plane cell table (extraction traces,
   dF/F, segmentation and cell classification)
2. a subject container with the subject data
3. a device container with the microscope information

## License

See the [LICENSE](LICENSE) file for details.

## Authors

Developed by the Allen Institute for Neural Dynamics (AIND) team.
