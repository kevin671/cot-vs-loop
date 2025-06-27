# satは難しいかもしれないけど
# uniform generation of 制約は何かしたいな...
# それとmcmcができると良いのかな？

# self redubilityと、あとはapproximate countingを手伝う確率的なアルゴリズムさえ実装できれば....
# loopdができないことを言うのは無理か。そもそもsampingができないから

# いやloopedでもcountingをやらせるんだった!!!!


class CountingFPRAS(task.GeneralizationTask):
    pass

class SamplingFPAUS(task.GeneralizationTask):
    pass