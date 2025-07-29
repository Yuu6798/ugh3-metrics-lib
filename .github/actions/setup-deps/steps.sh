#!/usr/bin/env bash
set -e
pip install -e .[dev]
pip install jellyfish>=1.0 vulture>=2.9
