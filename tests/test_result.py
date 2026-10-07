#!/usr/bin/env python3
from pathlib import Path

from verify_result import default_paths, verify


def test_feeder_se():
    verify(*default_paths())
