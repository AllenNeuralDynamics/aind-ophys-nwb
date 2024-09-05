# hash:sha256:ef64ff17fc6ca0c678f82140d5b19c8259e32919a509a4d099642e97efe36d36
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

COPY postInstall /
RUN /postInstall
