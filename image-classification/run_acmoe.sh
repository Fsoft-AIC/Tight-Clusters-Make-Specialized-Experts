CUDA_VISIBLE_DEVICES=0,2,3,4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 2282  --use_env main_moe.py \
--cfg configs/swincosa/R_swin_moe_small_patch4_window7_192_16expert_4gpu_1k.yaml --data-path /cm/shared/cuongdc10/datasets/image \
--batch-size 48 \
--output /cm/archive/stefannvkp/ellipswin/output \
--attack 'pgd' --eps 5 
#--compute-router-stability

#--use_wandb --project_name EllipSwin-spsa --job_name cosa2_TOP2_MAD_globscale_cpos357911131517 \
