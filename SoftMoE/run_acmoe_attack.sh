###### FOR TRAINING
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 8592 --nproc_per_node=4 --use_env main_train.py \
--model soft_acmoe_vit_tiny --data-path /cm/shared/cuongdc10/datasets/image --output_dir /cm/archive/stefannvkp/softmoe/output \
--job-name 'acmoe-7891011-mix8-bias0.35-clampthenscale' --moe-layer-index 6 7 8 9 10 11 --acmoe-layer-index 7 8 9 10 11 --mad --mix-weights --mix-k 8 \
--resume /cm/archive/stefannvkp/softmoe/output/acmoe-7891011-mix8-bias0.35-clampthenscale-bs128_checkpoint.pth \
--attack 'spsa' --batch-size 128 --eps 8 --eval
#--use-wandb 