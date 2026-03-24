1. 为什么要计算 compressed 和 images 的重建损失，这两是一样性质的吗

2. train_complete.py中包含了整个pipeline，考虑如何通过代码思考模型导入的方式
将模型分为Qwen部分和新添加部分，离散后通过PATOPipeline进行整合，那么PATO_model.py又如何定义呢

目前的问题：
    模型初始化问题
    一般使用完整、封装好的模型进行初始化，但是预训练模型参数又分开保存

分析封装好的PATO模型和Pipeline的优劣：
1 PATO模型需要保存庞大的预训练模型，毕竟模型不对Qwen部分进行训练，所以多此一举
2 Pipeline只需保存
