###### FOR TRAINING
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1117 --nproc_per_node=4 --use_env main_train.py \
--model soft_acmoe_vit_tiny --batch-size 128 --data-path /cm/shared/cuongdc10/datasets/image --output_dir /cm/archive/stefannvkp/softmoe/output \
--moe-layer-index 6 7 8 9 10 11 --acmoe-layer-index 7 8 9 10 11 --mad --mix-weights --mix-k 8 \