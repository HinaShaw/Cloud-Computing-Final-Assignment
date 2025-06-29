# Cloud-Computing-Final-Assignment
From Bocheng Xiao
项目概述
本项目聚焦于利用大模型技术进行多模态（情感、主题语义）谣言检测分析，通过结合情感语义分析、Twitter 主题建模以及多模态特征融合，实现对新闻文本的真假判别。项目涵盖以下核心模块：

基于大模型的情感语义分析，通过 Prompt 工程优化提升谣言检测准确率
基于 LDA 模型的 Twitter 主题分析，结合大模型进行主题语义深度解析
多模态特征融合的谣言综合预测模型，实现情感与主题语义的联合分析
技术架构
核心技术与工具
大模型：DeepSeek-R1:8B、ChatGLM3
主题模型：LDA（Latent Dirichlet Allocation）
特征融合：早期融合、跨模态注意力机制
可视化工具：pyLDAvis、词云图、热力图
向量检索：INSTRUCTOR 模型、faiss
环境依赖
python
运行
# 主要依赖库
pip install ollama
pip install gensim
pip install transformers
pip install numpy pandas
pip install matplotlib seaborn
pip install pyldavis
pip install wordcloud
pip install faiss-gpu
实验框架与方法
1. 基于大模型的情感语义分析
实验目标
利用大模型进行新闻真假判别与情感分析
通过 Prompt 优化提升谣言检测准确率
分析情感特征对真假判别任务的增益效果
关键步骤
本地部署 DeepSeek 或 GLM3 模型
设计多版本 Prompt 进行对比实验
统计三类准确率指标：
整体准确率（Accuracy）
假新闻准确率（Accuracy_fake）
真新闻准确率（Accuracy_true）
优化前后准确率对比
任务类型	旧 Prompt 准确率	新 Prompt 准确率	准确率提升
基础真假判别	0.5125	0.6720	+0.1595
融合情感分析	0.4980	0.7155	+0.2175
2. 基于大模型的 Twitter 主题分析
实验流程
数据预处理：分词、清洗、去停用词、词形还原
LDA 模型构建与训练
可视化分析：pyLDAvis 交互图、词云图、热力图
大模型深度解析主题语义
主题分析示例
主题 1：飓风桑迪相关的生活场景与个人经历
主题 2：灾害背景下的邻里互动与艺术元素
主题 3：城市灾害影响与地标建筑冲击
3. 多模态综合预测与分析
特征融合策略
早期融合：拼接主题特征与情感特征
注意力机制：动态对齐跨模态特征
模型架构
文本编码器：BERT
分类层：Softmax 二分类
性能指标
NDCG@10：0.82
MAP：0.78
单次查询时间：35ms
