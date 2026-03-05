# LLaMA-350M, Adam, 2 910b, 1 Node
# 'tb_linear_map'，'tb_sqrt'，'tb_log2'，'tb_step'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=35220 --master_addr=localhost scripts/train_llama.py \
  --config configs/llama_1024/llama-1B-t5.yaml \
  --optimizer.name=lars \
  --optimizer.learning_rate=0.001 \
  --optimizer.weight_decay=0.000001 \
  --swanlab.name=t5-1024-lars-1B-lr0_001-test \
  --save_folder=workspace/t5-1024-lars-1B-lr0_001-test \
#   --load_path=xxxx