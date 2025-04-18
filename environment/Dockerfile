# hash:sha256:deed82a22bccda3868240d700f2a1a18d6dbecb6b45afc713030487561e9db7b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update

# Install VIM
RUN apt-get install -y vim

RUN pip install -U --no-cache-dir \
    matplotlib \
    pandas \
    h5py \
    hdmf_zarr==0.11.0 \
	pynwb==2.8.3 \
    tifffile \
    imageio \
    aind-metadata-mapper \
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

