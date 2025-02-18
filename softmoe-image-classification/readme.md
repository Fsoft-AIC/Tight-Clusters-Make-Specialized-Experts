# SoftMoE Transformer with Adaptive Clustering (ACMoE) for Image Classification

This folder contains the implementation of ACMoE in the SoftMoE Transformer backbone for image classification under clean settings and under adversarial attack.

## Usage

### Install

We recommend using the pytorch docker `nvcr>=21.05` by
nvidia: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.

- Create a conda virtual environment and activate it:

```bash
conda create -n softmoe python=3.7 -y
conda activate softmoe
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch
```

- Install `timm==0.9.7`:

```bash
pip install timm==0.9.7
```

- Install other requirements:

### Training

Train ACMoE in the Tiny SoftMoE configuration using AC routing on the final half of transformer layers

```bash
CUDA_VISIBLE_DEVICES=<selected-gpus> python -m torch.distributed.launch --master_port 12345 --nproc_per_node=<num-gpus> --use_env main_train.py \
--model soft_acmoe_vit_tiny --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/checkpoints \
--moe-layer-index 6 7 8 9 10 11 --acmoe-layer-index 7 8 9 10 11 --mad --mix-weights --mix-k 8 \
```

Use the --moe-layer-index and --acmoe-layer-index flags to control placement of MoE and ACMoE layers.

### Evaluation under attack

Evaluate ACMoE under adversarial attacks using the --eval flag and selected attack

```bash
CUDA_VISIBLE_DEVICES=<selected-gpus> python -m torch.distributed.launch --master_port 12345 --nproc_per_node=<num-gpus> --use_env main_train.py \
--model soft_acmoe_vit_tiny --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/checkpoints \
--moe-layer-index 6 7 8 9 10 11 --acmoe-layer-index 7 8 9 10 11 --mad --mix-weights --mix-k 8 \
--resume /path/to/trained/checkpoint.pth \
--attack <selected-attack> --batch-size 128 --eps <perturbation-budget> --eval
```
