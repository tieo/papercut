#!/usr/bin/env sh
set -eu
nix-shell --run '.venv/bin/python docs/model/render_reports.py'
