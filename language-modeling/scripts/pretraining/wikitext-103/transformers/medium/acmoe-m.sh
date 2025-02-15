
args="
--data /fill/with/path \
--base_arch transformer \
--architecture sgsgscscscsc \
--gate_name smoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 1024 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 96 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--mu 0.2 \
--gamma 1.25 \
--checkpoint fill/with/path \
--use-var \
--mad 
"
echo "Training ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 7141 --nproc_per_node=4 --use_env train.py $args #--wandb-flag --project-name ellipSMoE


# echo "Eval ..."
# CUDA_VISIBLE_DEVICES='0,2,3,7' python -m torch.distributed.launch --master_port 1211 --nproc_per_node=4 --use_env train.py $args --resume --full-eval-mode --compute_load_balance
