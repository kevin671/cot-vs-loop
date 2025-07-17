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

### NC1
- Word Problem
- Boolean Formula
- **Arithmetic Expression**

Dataset generation
```shell
python gen_data/word.py --group=S5 --k=256 --data_dir=data/word_problem --samples=1000000 --overwrite
python gen_data/bfvp.py --max_depth 64 --data_dir=data/bfvp --train_size 1000000 --test_size 1000
python gen_data/arithmetic.py --max_depth 64 --train_size 1000000 --test_size 1000 --number_range 11 --under
```

Training
```shell
python -m experiments.train --task word --input_length 256 --model Looped --n_layer 2 --n_loop 8 --is_causal --epoch 1000
```
use --is_causal only for word problem

### NC2
- Reachability
- hoge
- hoge
- **Topological Sort**
- Regular Expression Matching
- Fixed Context-Free-Grammar Membership Testing 
- Pairwise Sequence Alignment (Longest Common Subsequence, **Edit Distnace**)

Dataset generation
```shell
python gen_data/path.py --num_nodes 8 --train_size 1000000 --test_size 1000 --data_dir data/path --seed 42

python gen_data/strings.py --train_size 1000000 --test_size 1000 --data_dir data/ed --length 16

python gen_data/strings.py --train_size 1000000 --test_size 1000 --data_dir data/lcs --length 16 --insert_cost 0 --delete_cost 0 --replace_cost 0 --match_cost 1 --objective max
```

Training
```shell
python -m experiments.train --task path --input_length 8 --model Looped --n_layer 2 --n_loop 8 --epoch 1000
python -m experiments.train --task path --input_length 8 --model Looped --n_layer 2 --is_causal--n_loop 8 --epoch 1000  
```

### P-complete
- Circuit Value Problem
- Iterated Mod 
- Unit Resolution

### #P
- Bayesian ?
Approximate inference in Bayesian networks.
Forward inference by ancestor sampling.

```shell
python gen_data/bayes_net.py
python experiments/train.py --task bayes_net --model GPT --n_embd 256 --n_head 4 --n_layer 2 --epoch 1000 
```

## Acknowledgement
- [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective (NeurIPS 2023)](https://github.com/guyuntian/CoT_benchmark)
- [Neural Networks and the Chomsky Hierarchy (ICLR 2023)](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main)
- [The Illusion of State in State-Space Models (ICML 2024)](https://github.com/jopetty/word-problem)
- [Progressive distillation induces an implicit curriculum (ICLR 2025)](https://github.com/abhishekpanigrahi1996/ProgressiveDistillation)