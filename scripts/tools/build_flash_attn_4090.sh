#!/bin/bash
# Build flash-attn 2 for the RTX 4090 container (fjl-habitat) without touching its Python.
#
# Why a build at all: the container runs torch 2.11.0+cu128, and the official
# flash-attn wheels stop at torch 2.8 (v2.8.3.post2), so a wheel for this torch has to
# be compiled here.  The container's own nvcc is CUDA 11.3, which cannot compile
# against torch's CUDA 12.8, and the pip package nvidia-cuda-nvcc-cu12 ships only
# ptxas/libnvvm (no nvcc driver, cicc or cudafe++), so a private CUDA 12.8.1 toolkit is
# put together from NVIDIA's redist archives instead.  cuBLAS/cuSPARSE/cuSOLVER/cuRAND
# headers come from torch's own cu12 wheels, which are the exact versions CUDA 12.8.1
# ships (12.8.4.1 / 12.5.8.93 / 11.7.3.90 / 10.3.9.90).
#
# Everything lands under $FA_BUILD (default /workspace/fa_build).  /opt/conda is only
# read: the result is a wheel in $FA_BUILD/dist and a private site dir $FA_BUILD/site,
# used by putting it on PYTHONPATH.  Installing the wheel into /opt/conda is a separate,
# deliberate step (see docs/ops/deploy_rtx4090.md).
#
# Run inside the container:  bash scripts/tools/build_flash_attn_4090.sh
# Env: FA_BUILD, MAX_JOBS (default 24), NVCC_THREADS (default 2).
set -euo pipefail

B=${FA_BUILD:-/workspace/fa_build}
PY=/opt/conda/bin/python
SP=$($PY -c "import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))")/nvidia
CH=$B/cuda-12.8
FA_VERSION=2.8.3.post1
FA_SHA256=55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348
PYPI=https://pypi.tuna.tsinghua.edu.cn
REDIST=https://developer.download.nvidia.cn/compute/cuda/redist

mkdir -p "$B/dl" "$B/unz" "$B/src" "$B/dist" "$B/tmp"

# --- 1. private CUDA 12.8.1 toolkit --------------------------------------------------
cd "$B/dl"
while read -r rel sha; do
  f=$(basename "$rel")
  [ -f "$f" ] || curl -sS -L --max-time 900 -o "$f" "$REDIST/$rel"
  echo "$sha  $f" | sha256sum -c -
done <<LIST
cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz 9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz 8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz 0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
cuda_nvtx/linux-x86_64/cuda_nvtx-linux-x86_64-12.8.90-archive.tar.xz 7755524026d30e2c211d793392166517e379724d19eb9e8e352693f7d5fa0c95
LIST
rm -rf "$B/unz" "$CH" && mkdir -p "$B/unz" "$CH"
for f in "$B"/dl/cuda_*-archive.tar.xz; do tar -xJf "$f" -C "$B/unz"; done
for d in "$B"/unz/*-archive; do cp -a "$d/." "$CH/"; done
rm -f "$CH/LICENSE"
for d in cublas cusparse cusolver curand; do cp -an "$SP/$d/include/." "$CH/include/"; done
rm -rf "$CH/include/__pycache__" "$CH/include/__init__.py"
# nvcc.profile links from lib64; the redist archives ship lib.
ln -sfn lib "$CH/lib64"
mkdir -p "$CH/targets/x86_64-linux"
ln -sfn ../../include "$CH/targets/x86_64-linux/include"
ln -sfn ../../lib "$CH/targets/x86_64-linux/lib"
"$CH/bin/nvcc" --version | tail -1

# --- 2. flash-attn from source -------------------------------------------------------
cd "$B/dl"
f=flash_attn-$FA_VERSION.tar.gz
if [ ! -f "$f" ]; then
  url=$(curl -sS --max-time 60 "$PYPI/simple/flash-attn/" | grep -oE "href=\"[^\"]*$f[^\"]*\"" | head -1 | sed -E 's/^href="//; s/"$//; s/#.*//')
  case $url in http*) ;; *) url="$PYPI/simple/flash-attn/$url" ;; esac
  curl -sS -L --max-time 900 -o "$f" "$url"
fi
echo "$FA_SHA256  $f" | sha256sum -c -
rm -rf "$B/src/flash_attn-$FA_VERSION"
tar -xzf "$f" -C "$B/src"

export CUDA_HOME=$CH
# ninja lives in /opt/conda/bin, which `docker exec` does not put on PATH; without it
# torch silently falls back to compiling the 73 units one at a time.
export PATH=$CH/bin:/opt/conda/bin:$PATH
export FLASH_ATTENTION_FORCE_BUILD=TRUE   # do not look for a prebuilt wheel on GitHub
export FLASH_ATTN_CUDA_ARCHS=80           # sm_80 SASS runs on sm_89, as the official wheels do
export MAX_JOBS=${MAX_JOBS:-24} NVCC_THREADS=${NVCC_THREADS:-2}
export TMPDIR=$B/tmp PIP_NO_CACHE_DIR=1
cd "$B/src/flash_attn-$FA_VERSION"
nice -n 19 $PY -m pip wheel . --no-build-isolation --no-deps -w "$B/dist" -v

rm -rf "$B/site"
$PY -m pip install --no-deps --no-compile --target "$B/site" "$B"/dist/flash_attn-"$FA_VERSION"-*.whl
echo "built: $(ls "$B"/dist/flash_attn-*.whl)"
echo "use:   PYTHONPATH=$B/site"
