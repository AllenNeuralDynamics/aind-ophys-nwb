# syntax=docker/dockerfile:1
FROM python:3.11-slim
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    matplotlib pandas h5py \
    hdmf==3.14.5 hdmf_zarr==0.11.0 pynwb==2.8.3 \
    tifffile imageio aind-nwb-utils==0.1.6 \
    nwbwidgets numpy==1.26.4 marshmallow sparse

RUN mkdir -p /data /results /scratch /schemas /raw /behavior /nwb /processed /capsule/code

COPY code/ /capsule/code/
