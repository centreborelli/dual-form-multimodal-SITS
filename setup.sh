#!/bin/bash

# Exit on error
set -e

pixi init
pip install -e .
pixi shell
pre-commit install
