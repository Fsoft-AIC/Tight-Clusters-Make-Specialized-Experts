###### FOR TRAINING
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1102 --nproc_per_node=4 --use_env main_train.py \
--model soft_moe_vit_tiny --batch-size 128 --data-path /cm/shared/cuongdc10/datasets/image --output_dir /cm/archive/stefannvkp/softmoe/output/baseline \
--moe-layer-index 6 7 8 9 10 11 \
--job-name 'softmoe-baseline-bs128' \
--resume /cm/archive/stefannvkp/softmoe/output/baseline/softmoe-baseline-bs128_checkpoint.pth \
--compute-router-stability --eval
#--use-wandb \



