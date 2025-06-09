#!/bin/bash
# Run the clean script first to ensure a clean environment
./clean.sh

# Prepare the domino assets
pip3 install httpx
python3 prepare_domino_assets.py | tee log.prepare_domino_assets

# Copy the initialConditions file from from_domino to 0_org/include
mkdir -p 0_org/include
cp from_domino/initialConditions 0_org/include/

# Copy the triSurface files as needed
mkdir -p constant/triSurface
cp from_domino/vehicle.stl constant/triSurface/

# Get the number of physical CPU cores (not including hyperthreading)
# For Ubuntu/Linux systems
procs=$(lscpu -p | grep -v '^#' | sort -u -t, -k 2,2 | wc -l)
echo "Detected physical cores: $procs"

# Update numberOfSubdomains in decomposeParDict to match available cores
sed -i "s/numberOfSubdomains [0-9]\+;/numberOfSubdomains $procs;/" system/decomposeParDict
echo "Updated decomposeParDict to use $procs cores."

# Prepare the 0 directory for the potentialFoam run
cp -r 0_org 0

### Mesh generation
surfaceFeatureExtract | tee log.surfaceFeatures
blockMesh | tee log.blockMesh
decomposePar
mpirun -np $procs --allow-run-as-root snappyHexMesh -parallel -overwrite | tee log.shm
mpirun -np $procs --allow-run-as-root checkMesh -parallel | tee log.checkmesh
reconstructParMesh -constant
rm -r processor*

### PotentialFoam run
decomposePar -force
mpirun -np $procs --allow-run-as-root potentialFoam -parallel -writephi -writePhi -writep | tee log.potentialfoam
reconstructPar -withZero
rm -r processor*

### Stitch together the potential flow and ML meshes using Python
cp -r 0/ 0_pf/
cp -r 0_org 0_hybrid
python3 make_hybrid_initialization.py | tee log.make_hybrid_initialization
cp -r 0_hybrid/ 0/

### PimpleFoam run
decomposePar -force
mpirun -np $procs --allow-run-as-root pimpleFoam -parallel | tee log.pimplefoam
reconstructPar -latestTime
rm -r processor*

### VTK output
foamToVTK -latestTime