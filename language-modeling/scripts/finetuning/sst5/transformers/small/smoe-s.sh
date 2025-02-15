
args="
--data /cm/shared/stefannvkp/language_modeling/text_finetune/sst5/ \
--data_name sst5 \
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
--lr 0.0001 \
--lr-warmup 0 \
--niter 5 \
--batch-sz 16 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint /cm/archive/stefannvkp/smoe_checkpoints/checkpoints/finetuning/sst5/medium/smoe-m-wikipre.pt
--pretrained_weight /cm/archive/stefannvkp/smoe_checkpoints/checkpoints/pretraining/wikitext103/smoe.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='6,7' python -m torch.distributed.launch --master_port 9195 --nproc_per_node=2 --use_env finetune_train.py $args

