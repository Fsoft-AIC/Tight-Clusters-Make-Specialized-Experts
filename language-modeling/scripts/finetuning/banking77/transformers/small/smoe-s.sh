args="
--data /cm/shared/stefannvkp/language_modeling/text_finetune/banking77 \
--data_name banking77 \
--base_arch transformer \
--architecture sgsgsgsgsgsg \
--gate_name smoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 1024 \
--dropout 0.1 \
--load_balance 0.00 \
--optim adam \
--lr 0.00001 \
--lr-warmup 0 \
--niter 50 \
--batch-sz 16 \
--batch-split 1 \
--nbatches 1000 \
--checkpoint /cm/archive/stefannvkp/smoe_checkpoints/checkpoints/finetune/banking77/transformers-m/smoe/smoe.pt \
--pretrained_weight /cm/archive/stefannvkp/smoe_checkpoints/checkpoints/pretraining/wikitext103/smoe.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 1066 --nproc_per_node=1 --use_env finetune_train2.py $args

