CUDA_VISIBLE_DEVICES=0,1,3,4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1945  --use_env main_moe.py \
--cfg configs/swincosa/R_swin_moe_small_patch4_window7_192_16expert_4gpu_1k_cosine.yaml --data-path /cm/shared/cuongdc10/datasets/image --batch-size 128 \
--use_wandb --project_name EllipSwin --job_name cosa2_top2_MAD_globscale_xmoeproj_experorthog_c357911131517
