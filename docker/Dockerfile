# Custom Ray image for the bufo sample — matches our dev stack on **Python 3.12**,
# eliminating the py3.9/version drift that made cluster jobs trial-and-error
# (datetime.UTC, zip(strict=), int.bit_count, torch.load CVE on .bin CLIP, ...).
#
# Design: bake the DEPS, ship the CODE per job. Project code (bufo/ + libs/) still
# arrives via `--working-dir .` + PYTHONPATH, so this image is stable across code
# changes — rebuild only when dependencies change.
#
# Base: Ray 2.40.0 (must match the cluster control plane) + py3.12 + CUDA.
# torch 2.4.1+cu121 is the build proven to run on the cluster's A100 driver; we keep
# it but on py3.12. (Bumping to local's torch 2.12 needs a cu wheel matching the node
# driver — verify before changing.)
FROM rayproject/ray:2.40.0-py312-gpu

RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Our third-party deps pinned to the repo's uv.lock (numpy<2 for torch 2.4.1;
# mlflow-skinny/protobuf pins keep Ray 2.40 happy — Cameron's guidance).
RUN pip install --no-cache-dir \
      diffusers==0.38.0 \
      transformers==5.11.0 \
      peft==0.19.1 \
      accelerate==1.13.0 \
      safetensors==0.8.0 \
      huggingface-hub==1.17.0 \
      tensorboard==2.20.0 \
      pyyaml==6.0.3 \
      "pydantic>=2.0" \
      pillow \
      "numpy<2" \
      "mlflow-skinny==2.22.0" \
      "protobuf<4"

# Sanity: imports must resolve on py3.12 (build fails loudly if a dep conflicts).
RUN python -c "import torch, diffusers, transformers, peft, accelerate, safetensors, tensorboard, pydantic, yaml, PIL; \
print('image stack OK:', 'torch', torch.__version__, '| transformers', transformers.__version__)"
