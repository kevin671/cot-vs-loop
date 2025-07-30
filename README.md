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

### NC1
- Word Problem

Dataset generation
```shell
python gen_data/word.py --group=S5 --k=256 --data_dir=data/word_problem --samples=1000000 --overwrite
```

Training
```shell
python -m experiments.train --task word --input_size 256 --model Looped --n_layer 2 --n_loop 8 --is_causal --epoch 1000
```

### TC1
- Arithmetic Expression
- Reachability

Dataset generation
```shell
python gen_data/arithmetic.py --max_depth 64 --train_size 10000000 --test_size 100000 --number_range 11 --under # 1M samples
python gen_data/path.py --num_nodes 32 --train_size 1000000 --test_size 100000 --data_dir data/path --seed 42
```

Training
```shell
python -m experiments.train --task arithmetic --input_size 32 --model Looped --n_loop 8 --epoch 1000
python -m experiments.train --task path --input_size 32 --model Looped --is_causal--n_loop 8 --epoch 1000
```

### NC2
- Context-free Grammar Recognition
- Pairwise Sequence Alignment (Longest Common Subsequence, Edit Distnace)

Dataset generation
```shell
python gen_data/cfg.py
python gen_data/strings.py --train_size 1000000 --test_size 100000 --data_dir data/ed --length 16
python gen_data/strings.py --train_size 1000000 --test_size 100000 --data_dir data/lcs --length 16 --insert_cost 0 --delete_cost 0 --replace_cost 0 --match_cost 1 --objective max
```

Training
```shell
python -m experiments.train --task cfg --input_size 32
python -m experiments.train --task lcs --input_size 64 --model TMLT --n_layer 1 --n_loop 16 --epoch 1000
python -m experiments.train --task ed --input_size 32 --model TMLT --n_layer 1 --is_causal--n_loop 8 --epoch 1000 
```

### #P
- Probabilistic Inference in Bayesian network

```shell
python gen_data/bayes_net.py
python experiments/train.py --task bayes_net --model GPT --n_embd 256 --n_head 4 --n_layer 2 --epoch 1000 
```

## Acknowledgement
- [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective (NeurIPS 2023)](https://github.com/guyuntian/CoT_benchmark)
- [Neural Networks and the Chomsky Hierarchy (ICLR 2023)](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main)
- [The Illusion of State in State-Space Models (ICML 2024)](https://github.com/jopetty/word-problem)
- [Physics of Language Models: Part 1, Learning Hierarchical Language Structures](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250639)