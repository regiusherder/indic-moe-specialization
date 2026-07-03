# Reproducible environment for a rented GPU pod (RunPod/Lambda, single
# 24GB GPU: RTX 4090/3090 class). Pin the base image tag — "latest" drifting
# silently is exactly the kind of untraceable-result risk this project needs
# to avoid.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Ubuntu 22.04's default python3 is 3.10 — that's what python3-pip targets
# and what actually runs; don't install a second interpreter that ENTRYPOINT
# would silently not use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Results are written to /workspace/results, which should be mounted as a
# volume (RunPod persistent volume / Lambda attached disk) so results survive
# past the pod's lifetime and aren't lost if the pod is terminated.
VOLUME ["/workspace/results"]

ENTRYPOINT ["python3", "scripts/run_all_models.py"]
