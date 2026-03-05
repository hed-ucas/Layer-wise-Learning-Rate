# LLaMA-350M, Adam, 2 910b, 1 Node
# 'tb_linear_map'，'tb_sqrt'，'tb_log2'，'tb_step'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost scripts/train_llama.py \
  --config configs/llama_1024/llama-1B-t5.yaml \
  --optimizer.name=adamw \
  --optimizer.learning_rate=0.0005 \
  --swanlab.name=t5-1024-adamw-1B-lr0_0005-test \
  --save_folder=workspace/t5-1024-adamw-1B-lr0_0005-test \
#   --load_path=workspace/t5-1024-LLR0-1B-lr0_0005-1-5/model_1000

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost scripts/train_llama.py \
  --config configs/llama_1024/llama-1B-t5.yaml \
  --optimizer.name=adamw \
  --optimizer.learning_rate=0.0025 \
  --swanlab.name=t5-1024-adamw-1B-lr0_0025-test \
  --save_folder=workspace/t5-1024-adamw-1B-lr0_0025-test \
#   --load_path=workspace/t5-1024-LLR0-1B-lr0_0005-1-5/model_1000