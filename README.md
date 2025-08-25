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
bash scripts/train.sh # train_chain.sh
```

### Deterministic Tasks that Admit Feasible Parallel Solutions
- Word Problem
- Graph Connectivity 
- Arithmetic Expression
- Edit Distance

Dataset generation:
```shell
python gen_data/word.py --group=S5 --k=64 --data_dir data/word_problem --samples=1000000
python gen_data/path.py --num_nodes 32 --train_size 1000000 --test_size 100000 --data_dir data/path --seed 42
python gen_data/arithmetic.py --max_depth 32 --train_size 1000000 --test_size 100000 --number_range 3 --under
python gen_data/strings.py --length 32 --train_size 1000000 --test_size 100000 --data_dir data/ed
```
Use `--chain` to generate a CoT dataset with intermediate steps following the full algorithms.


Training for Looped Transformers:
```shell
python -m experiments.train --task word --input_size 64 --model Looped --n_layer 2 --n_loop 8 --is_causal
python -m experiments.train --task path --input_size 32 --model Looped --n_layer 1 --n_loop 8
python -m experiments.train --task arithmetic --input_size 32 --model TMLT --n_layer 1 --n_loop 8
python -m experiments.train --task ed --input_size 32 --model TMLT --n_layer 1 --n_loop 8 --curriculum fixed_length
```
Replace with `--model GPT` and add `--chain` for training CoT.

### Probabilistic Inference

```shell
python experiments/train_bayes.py --task bayes_net --input_size 12 --model GPT --epoch 1 --seed 10 --chain --n_monte_carlo_samples 10 # CoT
python experiments/train_bayes.py --task bayes_net --input_size 12 --model Looped --is_causal --epoch 1 --seed 10 # Looped
python experiments/train_bayes.py --task bayes_net --deterministi --input_size 12 --model Looped --epoch 1 --seed 10 # deterministic control task for Looped
```

## Acknowledgement
- [Neural Networks and the Chomsky Hierarchy (ICLR 2023)](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main)
- [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective (NeurIPS 2023)](https://github.com/guyuntian/CoT_benchmark)
- [Why think step by step? Reasoning emerges from the locality of experience (NeurIPS 2023)](https://github.com/benpry/why-think-step-by-step)
- [The Illusion of State in State-Space Models (ICML 2024)](https://github.com/jopetty/word-problem)
