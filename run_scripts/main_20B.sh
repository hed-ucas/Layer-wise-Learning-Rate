#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# bash run_scripts/main_20B.sh

bash run_scripts/hed_scripts_1B_20B/adamw.sh
bash run_scripts/hed_scripts_1B_20B/lamb.sh
bash run_scripts/hed_scripts_1B_20B/lars.sh
bash run_scripts/hed_scripts_1B_20B/sharpness.sh
bash run_scripts/hed_scripts_1B_20B/LLR.sh