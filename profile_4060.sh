#!/bin/sh
# Run this on the machine that has the RTX 4060 and the CUDA toolkit.
# This Mac has neither nvcc nor an NVIDIA device, so the numbers are not collected here.
set -eu
cd "$(dirname "$0")"
mkdir -p runs
nvcc -O3 -arch=sm_89 -lineinfo src/stencil_27.cu -o runs/stencil_app -lcusparse
ncu --set full --export runs/profile_report ./runs/stencil_app
echo "Open runs/profile_report.ncu-rep in Nsight Compute, Details, Roofline."
