### acmoe robust ###
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 4517 --nproc_per_node=4 --use_env eval_OOD.py \
--data-path /cm/archive/stefannvkp/imnet-aorc --batch-size 128 --model soft_acmoe_vit_tiny \
--output_dir /cm/archive/stefannvkp/softmoe/output \
--resume /cm/archive/stefannvkp/softmoe/output/acmoe-7891011-mix8-bias0.35-clampthenscale-bs128_checkpoint.pth \
--eval --dist-eval \
--moe-layer-index 6 7 8 9 10 11 --acmoe-layer-index 7 8 9 10 11 --mad --mix-weights --mix-k 8 \
--which 'C' --use_wandb --job-name 'acmoe-7891011-evalbias0.35'


### baseline robust ###
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 4527 --nproc_per_node=4 --use_env eval_OOD.py \
# --data-path /cm/archive/stefannvkp/imnet-aorc --batch-size 128 --model soft_moe_vit_tiny \
# --output_dir /cm/archive/stefannvkp/softmoe/output/baseline \
# --resume /cm/archive/stefannvkp/softmoe/output/baseline/softmoe-baseline-bs128_checkpoint.pth \
# --eval --dist-eval \
# --moe-layer-index 6 7 8 9 10 11 \
# --which 'C' --use_wandb --job-name 'softmoe-baseline'