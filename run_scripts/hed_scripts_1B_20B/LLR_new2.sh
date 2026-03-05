# LLaMA-350M, Adam, 2 910b, 1 Node
# 'tb_linear_map'，'tb_sqrt'，'tb_log2'，'tb_step'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=35800 --master_addr=localhost scripts/train_llama.py \
  --config configs/llama_1024/llama-1B-t5.yaml \
  --optimizer.name=adamw \
  --LLR_ratio=1.0 \
  --optimizer.learning_rate=0.0005 \
  --LLR.use_modulewise_lr=True \
  --LLR.alpha_positively_with_lr=True \
  --LLR.unbalancedlr_every=100 \
  --LLR.grad_alpha_metric=grad \
  --LLR.num_grad_steps=0 \
  --LLR.grad_unbalancedlr_every=1 \
  --LLR.assign_func=tb_linear_map \
  --LLR.lr_min_ratio=1 \
  --LLR.lr_max_ratio=3 \
  --LLR.linear_steps=50 \
  --swanlab.name=t5-1024-LLR0-1B-lr0_0005-1-3 \
  --save_folder=workspace/t5-1024-LLR0-1B-lr0_0005-1-3 \
#   --load_path=