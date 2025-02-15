CUDA_VISIBLE_DEVICES='1,5,6,7' python -m torch.distributed.launch --master_port 4527 --nproc_per_node=4 --use_env eval_OOD.py \
--data-path /cm/archive/stefannvkp/imnet-aorc --batch-size 64 \
--output /cm/archive/stefannvkp/ellipswin/output \
--resume /cm/archive/stefannvkp/ellipswin/output/cosa_top2_c357911131517_epoch60/default/ckpt_epoch_60.pth.rank0 \
--cfg configs/swincosa/R_swin_moe_small_patch4_window7_192_16expert_4gpu_1k.yaml \
--eval --dist-eval