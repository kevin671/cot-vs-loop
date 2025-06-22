# To CoT or To Loop?

This repository provides the official implementation of the experiments in [To CoT or to Loop? A Formal Comparison Between Chain-of-Thought and Looped Transformers](https://arxiv.org/abs/2410.01405).

### Installation
hoge
```shell
conda create --name cotloop
conda activate cotloop
conda install pip
pip install -r requirements.txt
```

### Usage

```shell
conda activate nnch
export PYTHONPATH=$(pwd)/..
python experiments/train.py
```

全体のコードの設計

データセットは.txtに保存　（デバッグできるように）

examples以下に各データセットの例を表示する（わかりやすい）

訓練用のハイパラは何で指定しようか。各タスクごとに。実行スクリプトも特にいらないか。
train.pyを実行すれば（デフォルトで4層, 256, 4head、学習率1e-4とかは勝手に設定する

長さとLoopedの数を変えるのは面倒だな

Taskのapiが重要。必要な変数情報や.txtからモデルの入出力、評価関数が重要

各層にロスを入れてしまうか！これで正答率のグラフを書くのが一番良いかな？ (これの名前は...)
モデルはロスを返すようにするか

ミソはカリキュラム学習か...


## Looped Transfomrers and Parallel Computation

というか全部Loopedで解けたら嬉しいのか

入力が色々あるのがめんどくさい

グラフ、行列、文法などなど、、、

### P-complete

- The Circuit Value Problem
- Context-Free-Grammar Membership Testing 
- Linear Equalities （かいの判定）
- Iter Square? mod?

こっちはloopedが解けないor必要なループ数が著しく増えることが言えれば

それぞれにCOTを与えるのが結構めんどくさいけど

CoTはP-completeの方のみを

### NC
- CVP (NC) / graph reach? path?
- Regular / fixed CFG
- 逆行列の判定
- Arithmetic Expressio (Boolean Citrcuit) TC0で解けない


- perfect_matching (NC2) 本当に解けるのかな？


loopedの方で途中式を与えるには...？

教師を使いたい

CoTと違って何ができるかな

拡散モデルにみたいにpaddingして、学習する？まあこれが
もしくは途中の特徴

いや、どっちも嬉しくないな...

まあ実験としては良いのか。最終的な目標はカリキュラムだけど
→ ある意味でcompositionalの理論に展開されるのかな？

## CoT and Randomized Computation

ある意味でLoopeが解けない問題が解けたら素晴らしいな

BPP, FPRAS, 

確率的なCoTの学習はまだあんまりないかな？

## Acknowledgement
- [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective (NeurIPS 2023)](https://github.com/guyuntian/CoT_benchmark)
- [Neural Networks and the Chomsky Hierarchy (ICLR 2023)](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main)
