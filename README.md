# To CoT or to Loop?

This repository provides the official implementation of the experiments in [To CoT or to Loop? A Formal Comparison Between Chain-of-Thought and Looped Transformers](https://arxiv.org/abs/2410.01405).

### Installation
```shell
conda create --name cotloop
conda activate cotloop
pip install -r requirements.txt
```

### Usage

```shell
conda activate cotloop
python experiments/train.py
```

## Looped Transfomrers and Parallel Computation

### NC1
- Word Problem
- Boolean Formula
- Arithmetic Expression

Dataset generation
```shell
python gen_data/word.py --group=S5 --k=256 --data_dir=data/word_problem --samples=1000000 --overwrite
python gen_data/bfvp.py --length 16 --train_size 10 --test_size 10
python gen_data/arithmetic.py --length 256 --train_size 1000000 --test_size 100000 --number_range 11
```

Training
```shell
python -m experiments.train --task word --input_length 256 --model Looped --n_layer 2 --n_loop 8 --is_causal --epoch 1000
```
use --is_causal only for word problem

### NC2
- Reachability
- Fixed Context-Free-Grammar Membership Testing 

Dataset generation
```shell
python gen_data/path.py --num_nodes 8 --train_size 1000000 --test_size 1000 --data_dir data/path --seed 42
```

Training
```shell
python -m experiments.train --task path --input_length 8 --model Looped --n_layer 2 --n_loop 8 --epoch 1000

# CFGの方はcausalでも良いのかな？いやアルゴリズム的にだめな可能性も
python -m experiments.train --task path --input_length 8 --model Looped --n_layer 2 --is_causal--n_loop 8 --epoch 1000  
```

### P-complete
- Circuit Value Problem

Dataset generation
```shell
python gen_data/cvp.py --num_nodes 64 --train_size 1000000 --test_size 10000 --data_dir data/cvp --seed 42
python -m experiments.train --task cvp --input_length 64 --model Looped --n_layer 2 --n_loop 64 --is_causal --epoch 1000
```

### #P
- Bayesian ?
Approximate inference in Bayesian networks.
Forward inference by ancestor sampling.

vs. Fixed Circuit Value Problem?

```shell
python gen_data/bayes_net.py
python experiments/train.py --task bayes_net --model GPT --n_embd 256 --n_head 4 --n_layer 2 --epoch 1000 
```

## Acknowledgeme
- [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective (NeurIPS 2023)](https://github.com/guyuntian/CoT_benchmark)
- [Why think step by step? Reasoning emerges from the locality of experience (NeurIPS 2023)](https://github.com/benpry/why-think-step-by-step)
- [Neural Networks and the Chomsky Hierarchy (ICLR 2023)](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main)
- [The Illusion of State in State-Space Models (ICML 2024)](https://github.com/jopetty/word-problem)