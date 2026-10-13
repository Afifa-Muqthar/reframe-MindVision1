"""
Shared hardware auto-tuning helper. Several modules trade accuracy/latency
against peak RAM (model precision, slicing, whether a model stays resident
across requests); this centralizes the one RAM check they all make instead
of duplicating the threshold in each module.

The project's documented minimum target is an 8GB-RAM laptop (see
SETUP_GUIDE.md), with a 3.77GB machine as the original stress-test floor
(see image_gen/generator.py's resolution note). Below _AMPLE_RAM_GB, every
module keeps the conservative, low-memory behavior that floor required.
Above it (e.g. a 16GB dev machine), modules opt into the faster/higher
quality path instead.
"""

import psutil

_AMPLE_RAM_GB = 8


def has_ample_ram() -> bool:
    return psutil.virtual_memory().total / (1024 ** 3) >= _AMPLE_RAM_GB
