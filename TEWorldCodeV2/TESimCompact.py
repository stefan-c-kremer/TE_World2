#!/usr/bin/env python3
"""Run TE World 2 with the compact NumPy chromosome backend."""

from pathlib import Path

import numpy as np

from compact_backend import CompactTestChromosome2
import TESim


TESim.TestChromosome2 = CompactTestChromosome2
TESim.SIMULATION_BACKEND = "compact_numpy_v1"
TESim.ENTRYPOINT_FILE = str(Path(__file__).resolve())
TESim.BACKEND_FILE = str(Path(__file__).with_name("compact_backend.py").resolve())
TESim.BACKEND_RUNTIME = {"numpy_version": np.__version__}
TESim.CHECKPOINT_FORMAT = "pickle_gzip_v1"


if __name__ == "__main__":
    raise SystemExit(TESim.main())
