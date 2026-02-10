# StableQAT
This repository is the Pytorch implementation of **StableQAT**, a simple and effective QAT framework that stabilizes training in ultra low-bit settings via a novel, lightweight, and theoretically grounded surrogate for backpropagation derived from a discrete Fourier analysis of the rounding operator. StableQAT strictly generalizes STE as the latter arises as a special case of our more expressive surrogate family, yielding smooth, bounded, and inexpensive gradients that improve QAT training performance and stability across various hyperparameter choices. In experiments, StableQAT exhibits stable and efficient QAT at 2-4 bit regimes across Llamas and ViTs, demonstrating improved training stability and superior performance with negligible training overhead against other QAT techniques. 

(The current release supports models of the **Llama-3.2-1b** series.)

## Contents
- [Setup](#Setup)
- [Train](#Train)
- [Evaluation](#Evaluation)
- [Efficiency Comparison](#EfficiencyComparison)


## Setup
 ```
conda create -yn stableqat python=3.11
conda activate stableqat

pip install -r requirements.txt
 ```
   
## Train
### Step 1: Sample data
Sample from SlimPjama
```
python sample_data/sample_slim_pajama.py
```
Sample from Fineweb-edu
```
python sample_data/sample_fineweb_edu.py
```

### Step 2: Pre-tokenize
```
python pre_tokenize.py \
--tokenize_config_file $tokenize_config_file\
--train_data_dir $train_data_dir \
--train_data_file $train_data_file \
--tokenized_data_dir $tokenized_data_dir
```
Argument `$tokenize_config_file` refers to the file under the [tokenize_configs](./tokenize_configs) folder


### Step 3: Train
```
torchrun --nnodes=2 --nproc_per_node=8 train.py \
--train_config_file $train_config_file \
--train_data_dir $train_data_dir \
--tokenized_data_name $tokenized_data_dir \
--output_dir $output_dir
```
Argument `$train_config_file` refers to the file under the [train_configs](./train_configs) folder

### Evaluation
```
python -m lm_eval \
--model hf \
--tasks piqa,winogrande,arc_challenge,hellaswag,arc_easy,sciq,openbookqa,boolq \
--batch_size auto \
--model_args pretrained=$output_dir/last_checkpoint \
--device cuda:0 \
--output_path $output_dir/evaluation_results
```

## Efficiency Comparison
To compare the time cost of StableQAT, DSQ, ParetoQ for Llama-3.2-1B, run:
```
python efficiency_benchmark/compare_model_time_cost.py
```