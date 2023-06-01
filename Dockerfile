FROM nvcr.io/nvidia/pytorch:23.01-py3
#already has numpy, pyyaml, pillow

RUN apt-get update

RUN pip install h5py

WORKDIR /diffusion-asformer