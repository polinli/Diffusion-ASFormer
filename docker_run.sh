docker run -it \
    --shm-size 12G \
    --gpus all  \
    -v $(pwd):/diffusion-asformer \
    -v ~/paul/dataset/data:/diffusion-asformer/data \
    --name diffusion-asformer \
    paul/diffusion-asformer:latest \
    bash