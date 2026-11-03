"""Thin Code Ocean entry point for NWB packaging of ophys pipeline outputs.

All logic lives in the ``aind-ophys-nwb-library`` package; this wrapper only
parses settings (CLI / environment) and invokes ``run``.
"""

from aind_ophys_nwb_library.job import run

if __name__ == "__main__":
    run()
