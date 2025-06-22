import abc

# block sizeなども計算する
# interfaceも全部定める
# 評価ももちろん
# datasetがやるべきは
# task.prepare()を実行して
# 毎回ある行を読み込んで、ある処理に従って、xとyを返す
# 共通できる部分は共通化する

# データセットの生成も関数として定義されていると、呼び出せて楽かな？
# nを引数にする関数

class GeneralizationTask(abc.ABC):
    pass
