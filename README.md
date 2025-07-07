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
- **Arithmetic Expression** (これそもそもNC1なのか...？) CoTと比較

Word problemsは
n = 256まで (16, 32, 64, 128, 256)
ループ数は32とかまでで十分かな？

Boolean formulaは n=64まで
ループ数は16くらいで解けてほしい？

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
- Regular Expression Matching (本当にNC2?)
- Fixed Context-Free-Grammar Membership Testing 
- **pairwise sequence alignment (Longest Common Subsequence, Edit Distnace (本当にNC2?))** CoTと比較

n = 4, 8, 16, 32, 64くらいでよいかな？
ループ数が4, 9, 16, 25, 36的な感じで増えるといいけど
カリキュラムありなら n = 64までいけるかな？

動的計画法は
n = ... , 32, 64, ループ数64, 91くらいまで？
tmltを使う？論文で一言いうか、positional encodingを実装するための方法が2つあって、universalやtmlt的

カリキュラムを使う

Dataset generation
```shell
python gen_data/path.py --num_nodes 8 --train_size 1000000 --test_size 1000 --data_dir data/path --seed 42

python gen_data/strings.py --train_size 1000000 --test_size 1000 --data_dir data/ed --length 16

python gen_data/strings.py --train_size 1000000 --test_size 1000 --data_dir data/lcs --length 16 --insert_cost 0 --delete_cost 0 --replace_cost 0 --match_cost 1 --objective max
```
set length for {16,32,64}

chainのやつはn=32,64で、

Training
```shell
python -m experiments.train --task path --input_length 8 --model Looped --n_layer 2 --n_loop 8 --epoch 1000

# CFGの方はcausalでも良いのかな？いやアルゴリズム的にだめな可能性も
python -m experiments.train --task path --input_length 8 --model Looped --n_layer 2 --is_causal--n_loop 8 --epoch 1000  
```

### #P
- Bayesian ?
Approximate inference in Bayesian networks.
Forward inference by ancestor sampling.

vs. Fixed Circuit Value Problem?

これはP-completeなのか？まあとにかくLoopedで解けることを確認して、それと全く同じ学習方法で解けないことが言えれば

あとCoTで解けることをちゃんと確認する

```shell
python gen_data/bayes_net.py
python experiments/train.py --task bayes_net --model GPT --n_embd 256 --n_head 4 --n_layer 2 --epoch 1000 
```

## Acknowledgement
- [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective (NeurIPS 2023)](https://github.com/guyuntian/CoT_benchmark)
- [Neural Networks and the Chomsky Hierarchy (ICLR 2023)](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main)
- [The Illusion of State in State-Space Models (ICML 2024)](https://github.com/jopetty/word-problem)