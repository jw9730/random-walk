
# Random Walk Neural Networks - Llama 3 (PyTorch)

This directory contains the code for:

- Transductive classification on ogbn-arxiv (Section 5.3, Appendix A.6 and A.8)

- Transductive classification on Cora (20-shot), Cora, Citeseer, and Amazon Ratings (Appendix A.6)

## Setup

For Llama 3 experiments, we use a different environment from the rest of the experiments.

```bash
# setup docker container
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
docker run -it --gpus all --ipc host --name rw_l3 -v /home:/home pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime bash

# clone repository and compile graph-walker
git clone https://github.com/jw9730/random-walk.git random-walk
cd random-walk
bash install_walker.sh
cd src_llama3

# install dependencies
pip3 install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip3 install -r requirements.txt

# prepare llama 3 checkpoints
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ../experiments/checkpoints/Meta-Llama-3-8B-Instruct-HF
huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --local-dir ../experiments/checkpoints/Meta-Llama-3-70B-Instruct-HF

# prepare environment variables
export LLAMA3_8B_INSTRUCT_HF='../experiments/checkpoints/Meta-Llama-3-8B-Instruct-HF'
export LLAMA3_70B_INSTRUCT_HF='../experiments/checkpoints/Meta-Llama-3-70B-Instruct-HF'
export SAVE_DIR='../experiments/llama3'
export VLLM_WORKER_MULTIPROC_METHOD='spawn'
```

## Experiments

### Training-free transductive classification (Section 5.3, Appendix A.6 and A.8)

Model inputs and predictions can be found at [this link](https://drive.google.com/drive/folders/1kZbmjRkrRnhhmhex4vbMnpXMP6WCZtRP?usp=sharing).
To evaluate the accuracy:

1. Find the file `[DATASET]/[EXP_NAME].tar.gz` of interest.
2. Download the file into `../experiments/llama3/[DATASET]/[EXP_NAME].tar.gz` and extract it.

After then, you can bypass running model predictions and run the accuracy evaluation directly.

ogbn-arxiv (Section 5.3, Appendix A.8)

```bash
# RWNN
# predictions
python3 main_arxiv.py --save_name 8b_walk --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_arxiv.py --save_name 8b_walk_val --alpha 0.7 --no_backtrack --include_neighbors --use_val_labels --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_arxiv.py --save_name 70b_walk_val --alpha 0.7 --no_backtrack --include_neighbors --use_val_labels --ckpt_dir $LLAMA3_70B_INSTRUCT_HF
python3 main_arxiv.py --save_name 70b_walk_val_a0.3 --alpha 0.3 --no_backtrack --include_neighbors --use_val_labels --ckpt_dir $LLAMA3_70B_INSTRUCT_HF
# evaluation
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/8b_walk
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/8b_walk_val
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/70b_walk_val
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/70b_walk_val_a0.3

# zero-shot baselines
# predictions
python3 main_baselines_arxiv.py --save_name 8b_zero_shot --n_shots 0 --n_preds 1 --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_baselines_arxiv.py --save_name 70b_zero_shot --n_shots 0 --n_preds 1 --ckpt_dir $LLAMA3_70B_INSTRUCT_HF
# evaluation
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/8b_zero_shot --n_preds 1
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/70b_zero_shot --n_preds 1

# one-shot baselines
# predictions
python3 main_baselines_arxiv.py --save_name 8b_one_shot --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_baselines_arxiv.py --save_name 8b_one_shot_val --use_val_labels --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_baselines_arxiv.py --save_name 70b_one_shot --ckpt_dir  $LLAMA3_70B_INSTRUCT_HF
python3 main_baselines_arxiv.py --save_name 70b_one_shot_val --use_val_labels --ckpt_dir  $LLAMA3_70B_INSTRUCT_HF
# evaluation
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/8b_one_shot
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/8b_one_shot_val
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/70b_one_shot
python3 parse_results_arxiv.py --save_dir $SAVE_DIR/arxiv/70b_one_shot_val

# visualization
torchrun --nproc_per_node 1 --master_port 1000 main_arxiv_attention_vis.py --load_dir ../experiments/llama3/arxiv/8b_walk_val --save_dir ../experiments/llama3/arxiv/8b_walk_val_vis --max_batch_size 1 --ckpt_dir $LLAMA3_8B_INSTRUCT/original --tokenizer_path $LLAMA3_8B_INSTRUCT_HF/original/tokenizer.model
# after running the above, run attention_vis_arxiv.ipynb
```

Cora (20-shot), Cora, Citeseer, and Amazon Ratings (Appendix A.6)

```bash
# Cora 20-shot
# predictions
python3 main_cora_20shot.py --save_name 8b_walk --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_cora_20shot.py --save_name 8b_walk_val --alpha 0.7 --no_backtrack --include_neighbors --use_val_labels --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
# evaluation
python3 parse_results_cora_20shot.py --save_dir $SAVE_DIR/cora_20shot/8b_walk
python3 parse_results_cora_20shot.py --save_dir $SAVE_DIR/cora_20shot/8b_walk_val

# Cora
# predictions
python3 main_cora.py --save_name 8b_walk --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_cora.py --save_name 8b_walk_val --alpha 0.7 --no_backtrack --include_neighbors --use_val_labels --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
# evaluation
python3 parse_results_cora.py --save_dir $SAVE_DIR/cora/8b_walk
python3 parse_results_cora.py --save_dir $SAVE_DIR/cora/8b_walk_val

# Citeseer
# predictions
python3 main_citeseer.py --save_name 8b_walk --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_citeseer.py --save_name 8b_walk_val --alpha 0.7 --no_backtrack --include_neighbors --use_val_labels --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
# evaluation
python3 parse_results_citeseer.py --save_dir $SAVE_DIR/citeseer/8b_walk
python3 parse_results_citeseer.py --save_dir $SAVE_DIR/citeseer/8b_walk_val

# Amazon Ratings
# predictions
python3 main_amazonratings.py --save_name 8b_walk --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
python3 main_amazonratings.py --save_name 8b_walk_val --alpha 0.7 --no_backtrack --include_neighbors --use_val_labels --ckpt_dir $LLAMA3_8B_INSTRUCT_HF
# evaluation
python3 parse_results_amazonratings.py --save_dir $SAVE_DIR/amazonratings/8b_walk
python3 parse_results_amazonratings.py --save_dir $SAVE_DIR/amazonratings/8b_walk_val
```

### Fine-tuning (Appendix A.6)

Fine-tuned model checkpoints can be found at [this link](https://drive.google.com/drive/folders/1p6aoP-yqDixx2e-wn-kMVyy5ID1jF_Py?usp=sharing).
We provide the checkpoints for Citeseer and Amazon Ratings; unfortunately, checkpoints for arXiv, Cora (20-shot), and Cora are lost.
We still provide the model inputs and predictions for all datasets, so that their accuracies can be evaluated.

To load a checkpoint:

1. Find the directory `[DATASET]/checkpoint-[EPOCH]` of interest.
2. Download it into  `../experiments/llama3/finetuning_checkpoints/[DATASET]/checkpoint-[EPOCH]`.

After then, you can bypass fine-tuning and run the prediction and accuracy evaluation directly.

Model inputs and predictions can be found at [this link](https://drive.google.com/drive/folders/1o0jFWgg0c74qIot7B6Fd-CUF_UzGFtzV?usp=sharing).
To evaluate the accuracy:

1. Find the file `[DATASET].tar.gz` of interest.
2. Download the file into `../experiments/llama3/finetuning_test_outputs/[DATASET].tar.gz` and extract it.

After then, you can bypass running model predictions and run the accuracy evaluation directly.

Prepare dependencies and environment variables

```bash
# https://www.philschmid.de/fsdp-qlora-llama3
pip3 install datasets==2.18.0 evaluate==0.4.1 bitsandbytes==0.43.2 trl==0.8.6 peft==0.10.0 tensorboardX
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export CKPT_DIR='../experiments/llama3/finetuning_checkpoints'
export FT_SAVE_DIR='../experiments/llama3/finetuning_test_outputs'
```

Make datasets

```bash
cd finetuning
python3 makedata_arxiv.py --alpha 0.7 --no_backtrack --include_neighbors --tokenizer_path ../$LLAMA3_8B_INSTRUCT_HF/original/tokenizer.model
python3 makedata_cora_20shot.py --alpha 0.7 --no_backtrack --include_neighbors --tokenizer_path ../$LLAMA3_8B_INSTRUCT_HF/original/tokenizer.model
python3 makedata_cora.py --alpha 0.7 --no_backtrack --include_neighbors --tokenizer_path ../$LLAMA3_8B_INSTRUCT_HF/original/tokenizer.model
python3 makedata_citeseer.py --alpha 0.7 --no_backtrack --include_neighbors --tokenizer_path ../$LLAMA3_8B_INSTRUCT_HF/original/tokenizer.model
python3 makedata_amazonratings.py --alpha 0.3 --no_backtrack --include_neighbors --tokenizer_path ../$LLAMA3_8B_INSTRUCT_HF/original/tokenizer.model
cd ..
```

Run fine-tuning

```bash
cd finetuning
torchrun --nproc_per_node=1 --master_port 1000 fsdp_qlora.py --config configs/arxiv.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 1000 fsdp_qlora.py --config configs/cora_20shot.yaml
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 1001 fsdp_qlora.py --config configs/cora.yaml
torchrun --nproc_per_node=1 --master_port 1000 fsdp_qlora.py --config configs/citeseer.yaml
torchrun --nproc_per_node=1 --master_port 1000 fsdp_qlora.py --config configs/amazonratings.yaml
cd ..
```

Run evaluation

```bash
# predictions
cd finetuning
torchrun --nproc_per_node 1 --master_port 1000 eval_arxiv.py --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir ../$CKPT_DIR/arxiv/checkpoint-4600 --save_dir ../$FT_SAVE_DIR/arxiv
torchrun --nproc_per_node 1 --master_port 1000 eval_cora_20shot.py --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir ../$CKPT_DIR/cora_20shot/checkpoint-1200 --save_dir ../$FT_SAVE_DIR/cora_20shot
torchrun --nproc_per_node 1 --master_port 1000 eval_cora.py --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir ../$CKPT_DIR/cora/checkpoint-1400 --save_dir ../$FT_SAVE_DIR/cora
torchrun --nproc_per_node 1 --master_port 1000 eval_citeseer.py --n_preds 5 --alpha 0.7 --no_backtrack --include_neighbors --ckpt_dir ../$CKPT_DIR/citeseer/checkpoint-800 --save_dir ../$FT_SAVE_DIR/citeseer
torchrun --nproc_per_node 1 --master_port 1000 eval_amazonratings.py --alpha 0.3 --no_backtrack --include_neighbors --ckpt_dir ../$CKPT_DIR/amazonratings/checkpoint-600 --save_dir ../$FT_SAVE_DIR/amazonratings
cd ..

# evaluation
python3 parse_results_arxiv.py --save_dir $FT_SAVE_DIR/arxiv --n_preds 1
python3 parse_results_cora_20shot.py --save_dir $FT_SAVE_DIR/cora_20shot --n_preds 1
python3 parse_results_cora.py --save_dir $FT_SAVE_DIR/cora --n_preds 1
python3 parse_results_citeseer.py --save_dir $FT_SAVE_DIR/citeseer --n_preds 5
python3 parse_results_amazonratings.py --save_dir $FT_SAVE_DIR/amazonratings --n_preds 1
```
