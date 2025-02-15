args="
--data /cm/shared/stefannvkp/language_modeling/wikitext-103 \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsfsgsfsgsf \
--gate_name xmoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.00007 \
--lr-warmup 4000 \
--niter 120 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint /cm/archive/stefannvkp/smoe_checkpoints/checkpoints/pretraining/wikitext103/glam-xmoe-m \
--distributed
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 7141 --nproc_per_node=4 --use_env train.py $args --resume --wandb-flag --project-name ellipSMoE


# echo "Eval ..."
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1211 --nproc_per_node=4 --use_env train.py $args --resume --full-eval-mode
