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

### NC

NC1
- Word Problem
- Boolean formula value problem

NC2
- Reachability
- Linear Equalition
- Fixed Context-Free-Grammar Membership Testing 

Dataset generation
```shell
python data_gen/word.py --group=S5 --k=512 --samples=1000000 --overwrite
python data_gen/arithmetic.py --length 512 --train_size 1e6 --test_size 1e5 --number_range 11
```

### P-complete

- Circuit Value Problem
- Linear Equalition
- Arithmetic Expression

### #P
Approximate inference in Bayesian networks.
Forward inference by ancestor sampling.

## Acknowledgement
- [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective (NeurIPS 2023)](https://github.com/guyuntian/CoT_benchmark)
- [Why think step by step? Reasoning emerges from the locality of experience ((NeurIPS 2023))](https://github.com/benpry/why-think-step-by-step)
- [Neural Networks and the Chomsky Hierarchy (ICLR 2023)](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main)
- [The Illusion of State in State-Space Models (ICML 2024)](https://github.com/jopetty/word-problem)