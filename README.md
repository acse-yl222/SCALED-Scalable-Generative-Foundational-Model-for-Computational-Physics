# SCALED : SCALable gEnerative founDational model for Computational Physics
SCALED is a scalable foundational neural network built upon a diffusion-based generative framework for computational physics.

## Demo demostration
### Urban Flow

## Building up the environment
```bash
conda create -n scaled python=3.10
###
pip install torch torchvision torchaudio
pip install -r requirement.txt
pip install -e .
```

## Download the building geometry
1. South Kensington
2. Generated Area
3. Cubiod

## Generated Dataset
1. South Kensington
```
python tools/dataset_generator/south_kensington.py
```
2. Generated Area
```
python tools/dataset_generator/generated_area.py
```
3. Cubiod
```
python tools/dataset_generator/cuboid.py
```

## Trainning
The code support multi-gpus trainning
Multi-gpus trainning, using 4 gpus.
```bash
accelerate launch --num_processes 4 tools/trainning_sfc/trainning_stage1.py
```
single-gpus trainning, using 1 gpus.
```bash
python tools/trainning_sfc/trainning_stage2.py
```

## Inference
The code support multi-gpus inference
```bash
accelerate launch --num_processes 4 tools/trainning_sfc/trainning_stage1.py
```