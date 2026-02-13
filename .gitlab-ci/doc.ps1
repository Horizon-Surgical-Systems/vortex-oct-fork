# setup conda (ref: https://stackoverflow.com/questions/57754356/activating-conda-environment-during-gitlab-ci)
conda init powershell | Out-Null
if($LastExitCode -ne 0) { exit $LastExitCode }
if (Test-Path $PROFILE.CurrentUserAllHosts) {
    & $PROFILE.CurrentUserAllHosts
}

$PY_VER_NODOT = $PY_VER -Replace '\.', ''
$CUDA_VER_NODOT = $CUDA_VER -Replace '\.', ''

# setup conda environment
conda create --name $VENV python=$PY_VER numpy -q -y
if($LastExitCode -ne 0) { exit $LastExitCode }
conda activate $VENV
if($LastExitCode -ne 0) { exit $LastExitCode }
conda install --channel conda-forge doxygen
if($LastExitCode -ne 0) { exit $LastExitCode }

# install non-conda dependencies
Write-Host "-- Installing doc requirements"
& $Python -m pip install -r .gitlab-ci/requirements-doc.txt
if($LastExitCode -ne 0) { exit $LastExitCode }

# install Vortex
Get-ChildItem build/vortex/dist
$Vortex = Join-Path build/vortex/dist "vortex_cuda${CUDA_VER_NODOT}-*-*${PY_VER_NODOT}*-*${PY_VER_NODOT}*-win_${ARCH}.whl" -Resolve
if(!($Vortex)) {
    Write-Host "Could not find Vortex wheel"
    exit 1
}
Write-Host "-- Installing Vortex from $Vortex"
& $Python -m pip install $Vortex
if($LastExitCode -ne 0) { exit $LastExitCode }

# run doxygen for C++ auto-generation
doxygen build/vortex/doc/Doxyfile
if($LastExitCode -ne 0) { exit $LastExitCode }

# build the documentation
sphinx-build -b dirhtml -c build/vortex/doc/_sphinx -d build/vortex/doc/_sphinx doc build/vortex/doc/html
if($LastExitCode -ne 0) { exit $LastExitCode }
