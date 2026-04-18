# hash:sha256:f021ab0082d32d2954b63da31e988d47a581ca94879eb15b3627e51ed45c93a3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-ask-pass /

RUN apt-get update

# Install VIM
RUN apt-get install -y vim

RUN pip install -U --no-cache-dir \
    matplotlib \
    pandas \
    h5py \
    hdmf \
    git+https://github.com/hdmf-dev/hdmf-zarr.git@zarr-v3-migration \
	pynwb==2.8.3 \
    tifffile \
    imageio \
    aind-behavior-utils \
    aind-nwb-utils==0.1.6\
    nwbwidgets \
    numpy==1.26.4 \
    marshmallow \
    sparse

# install versions for github copilot
ADD "https://github.com/coder/code-server/releases/download/v4.21.1/code-server-4.21.1-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.21.1-linux-amd64/bin/code-server  /usr/bin/code-server

COPY postInstall /
RUN /postInstall

