# DAGを生成して、その評価を行う
# DAGの深さで場合分けする
# each gate is represented by four consecutive tokens, which are gate type, two input gate ids, and the current gate id.

# ついでにグラフ系の問題も同じように計算できるかな？

# とりあえず問題を作って、まあ当然Loopedで解けないという結論？
# いや逆か深さに比例して解けて欲しいのか...解けるかな