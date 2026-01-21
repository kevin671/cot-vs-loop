# Chain of thought vs. latent thought

This repository provides the official implementation of [A Formal Comparison Between Chain-of-Thought and Latent Thought](https://arxiv.org/abs/2505.19245).


## Installation
```shell
conda create --name cotloop
conda activate cotloop
pip install -r requirements.txt
conda activate cotloop
```

## Experiments

regular curiculm for looped

### Word Problem

Generate dataset.
```shell
python gen_data/word.py --group=S5 --k=64 --data_dir data/word_problem --samples=1000000
```

Train and evaluate.
```shell
python -m experiments.train --task word --input_size 64 --model GPT --n_layer 2 --chain
python -m experiments.train --task word --input_size 64 --model Looped --n_layer 2 --n_loop 8 --is_causal
```

### Graph Connectivity 

Generate dataset.
```shell
python gen_data/path.py --num_nodes 16 --train_size 1000000 --test_size 100000 --data_dir data/path --seed 42 --make_chain
python gen_data/path.py --num_nodes 32 --train_size 1000000 --test_size 100000 --data_dir data/path --seed 42
```

Train and evaluate.
```shell
python -m experiments.train --task path --input_size 16 --model GPT --n_layer 2 -chain
python -m experiments.train --task path --input_size 32 --model Looped --n_layer 2 --n_loop 8
```

### Arithmetic Expression

Generate dataset.
```shell
python gen_data/arithmetic.py --max_depth 16 --train_size 1000000 --test_size 100000 --number_range 3 --under --make_chain
python gen_data/arithmetic.py --max_depth 32 --train_size 1000000 --test_size 100000 --number_range 3 --under
```

Train and evaluate.
```shell
python -m experiments.train --task arithmetic --input_size 16 --model GPT --n_layer 2 --chain
python -m experiments.train --task arithmetic --input_size 32 --model TMLT --n_layer 1 --n_loop 8 --curriculum regular
```

### Edit Distance
```shell
python gen_data/strings.py --length 16 --train_size 1000 --test_size 100 --data_dir data/ed --make_chain
python gen_data/strings.py --length 32 --train_size 1000000 --test_size 100000 --data_dir data/ed
```

Train and evaluate.
```shell
python -m experiments.train --task ed --input_size 16 --model GPT --n_layer 2 --chain
python -m experiments.train --task ed --input_size 32 --model TMLT --n_layer 1 --n_loop 8 --curriculum fixed_length
```

For stepwise internalization.
```shell
python -m experiments.train_distill --task ed --input_size 16 --model GPT --n_layer 2 --epochs_per_stage 16 --remove_per_stage 8 \
```

###  Counting

### Approximate Counting of DNF

Generate dataset.
```shell
python gen_data/dnf.py
```

Train and evaluate.
```shell
python -m experiments.train_counting --task dnf --model GPT --chain --steps_per_epoch 10000 --num_mc_samples 1000
python -m experiments.train_counting --task dnf --model Looped --epoch 10 n_loop 100
```

### Approximate Sampling of Graph Coloring
Train and evaluate.
```shell
python -m experiments.train_counting --task coloring --model GPT --coloring_mcmc_steps 10 --chain
python -m experiments.train_counting --task coloring --model Looped n_loop 30
```


## Acknowledgement
- [Leveraging Neural Networks for Approximate DNF Counting (AAAI 2020)](https://github.com/ralphabb/NeuralDNF)
- [Neural Networks and the Chomsky Hierarchy (ICLR 2023)](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main)
- [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective (NeurIPS 2023)](https://github.com/guyuntian/CoT_benchmark)
- [The Illusion of State in State-Space Models (ICML 2024)](https://github.com/jopetty/word-problem)
- [Coconut: Training Large Language Models to Reason in a Continuous Latent Space (COLM 2025)](https://github.com/facebookresearch/coconut)
- [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step (2024)](https://github.com/da03/Internalize_CoT_Step_by_Step)