args="
--data /cm/shared/stefannvkp/language_modeling/text_finetune/sst2/ \
--data_name sst2 \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsfsgsfsgsf \
--gate_name smoe \
--nlayers 6 \
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
--batch-sz 16 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint /cm/archive/stefannvkp/smoe_checkpoints/checkpoints/finetuning/sst2/glam/medium/glam-smoe-m.pt
--pretrained_weight /cm/archive/stefannvkp/smoe_checkpoints/checkpoints/pretraining/wikitext103/glam-m.pt \
--distributed \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1995 --nproc_per_node=4 --use_env finetune_train.py $args


# echo "Eval ..."
# CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.launch --master_port 1903 --nproc_per_node=1 --use_env train.py $args --resume --full-eval-mode
