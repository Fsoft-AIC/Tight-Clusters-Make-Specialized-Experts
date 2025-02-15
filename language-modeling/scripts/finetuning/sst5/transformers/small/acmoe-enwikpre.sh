#mkdir -p checkpoints/enwik8/transformers-m/smoe

args="
--data fill/with/path \
--data_name sst5 \
--base_arch transformer \
--architecture sgsgscscscscscsc \
--gate_name smoe \
--nlayers 8 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.00 \
--optim adam \
--lr 0.0001 \
--lr-warmup 0 \
--niter 5 \
--batch-sz 8 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint fill/with/path
--pretrained_weight fill/with/path \
--use-var \
--mad \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='0,4,5,6' python -m torch.distributed.launch --master_port 1273 --nproc_per_node=4 --use_env finetune_train.py $args #--wandb-flag --project-name ellipSMoE


# echo "Eval ..."
# CUDA_VISIBLE_DEVICES='5,6,7' python -m torch.distributed.launch --master_port 1903 --nproc_per_node=3 --use_env train.py $args --resume --full-eval-mode
