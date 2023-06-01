docker run -it \
    --shm-size 16G \
    --gpus all  \
    -v $(pwd):/diffusion-asformer \
    -v ~/paul/dataset/:/diffusion-asformer/data/ \
    -v ~/paul/dataset/hdd_data/:/diffusion-asformer/data/hdd_data \
    --name diffusion-asformer \
    paul/diffusion-asformer:latest \
    bash