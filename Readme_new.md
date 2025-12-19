conda的python好像是3.12环境下，

构建环境只需要install requirement.txt
之后还要install spacy，并安装其大型英语模型等，spacy出现问题大概率是numpy版本与spacy版本等问题

使用前需要先在pretreatment中使用word_rele（代码内使用的绝对路径，需要改代码内的路径，不出意外是270行后的两个（不出意外的话））

所有逻辑基本集中在models\components\Sublayer.py的PairMultiHeadAttention中，里面存在部分废案代码，请以实际forword内为准（部分函数名不反应真实意义，属于历史遗产屎山代码）

另一部分存在于models\Predictor\的pred_tag中，不过其仅仅只是两层，预测结果是在PairMultiHeadAttention中计算的