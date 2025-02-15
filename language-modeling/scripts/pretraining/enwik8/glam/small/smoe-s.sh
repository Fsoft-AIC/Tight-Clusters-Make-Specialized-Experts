args="
--data /cm/shared/stefannvkp/language_modeling/enwik8 \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsf \
--gate_name smoe \
--nlayers 8 \
--hid-sz 264 \
--inner-hid-sz 264 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint /cm/archive/stefannvkp/smoe_checkpoints/checkpoints/enwik8/glam-s/glam-smoe-s.pt \
--distributed \
"

# echo "Training ..."
# CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1234 --nproc_per_node=4 --use_env train.py $args --wandb-flag --project-name ellipSMoE


echo "Eval ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1234 --nproc_per_node=1 --use_env train.py $args --resume --full-eval-mode
