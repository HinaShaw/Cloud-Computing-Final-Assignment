import re
import random
from pathlib import Path
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
import ollama
import asyncio
from typing import List, Dict, Tuple

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# 1. 数据准备
def load_and_sample_data(file_path, sample_size=10):
    """
    加载数据并随机抽取指定数量的样本
    """
    try:
        # 读取数据文件 - 使用新版本的pandas参数
        df = pd.read_csv(
            file_path,
            sep='\t',
            quoting=3,
            on_bad_lines='skip',  # 替换原来的error_bad_lines
            engine='python'  # 使用python引擎确保兼容性
        )

        # 随机抽取样本
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)

        # 返回文本列表
        return df['post_text'].tolist()
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


# 2. 数据预处理
def preprocess_text(texts):
    """
    文本预处理：清洗、分词、去停用词、词形还原
    """
    # 初始化工具
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_texts = []
    for text in texts:
        if not isinstance(text, str):
            continue

        # 清洗文本：去除非字母字符并转为小写
        text = re.sub(r'[^a-zA-Z]', ' ', text).lower()

        # 分词
        words = word_tokenize(text)

        # 去停用词和短词
        words = [word for word in words if word not in stop_words and len(word) > 2]

        # 词形还原
        words = [lemmatizer.lemmatize(word) for word in words]

        processed_texts.append(words)

    return processed_texts


# 3. 模型构建与训练
def train_lda_model(processed_texts, num_topics=3, passes=15):
    """
    训练LDA主题模型
    """
    # 创建词典
    dictionary = corpora.Dictionary(processed_texts)

    # 过滤极端值
    dictionary.filter_extremes(no_below=1, no_above=0.7)

    # 创建词袋模型
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # 训练LDA模型
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )

    return lda_model, corpus, dictionary


# 4. 可视化分析
def visualize_lda(lda_model, corpus, dictionary):
    """
    使用pyLDAvis可视化LDA模型
    """
    try:
        vis = gensimvis.prepare(lda_model, corpus, dictionary)

        # 尝试在Jupyter笔记本中显示
        try:
            from IPython.display import display
            display(vis)
        except ImportError:
            # 如果不在Jupyter环境中，保存为HTML文件
            output_file = "lda_visualization.html"
            pyLDAvis.save_html(vis, output_file)
            print(f"可视化已保存为 {output_file}，请在浏览器中打开查看")

        return vis
    except Exception as e:
        print(f"可视化失败: {e}")
        return None


def generate_wordclouds(lda_model, num_topics):
    """
    为每个主题生成词云图
    """
    for topic_id in range(num_topics):
        # 获取主题的词分布
        word_weights = lda_model.show_topic(topic_id, topn=20)

        # 创建词频字典
        word_freq = {word: weight for word, weight in word_weights}

        # 生成词云
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate_from_frequencies(word_freq)

        # 显示词云
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Topic {topic_id + 1} Keywords")
        plt.axis('off')
        plt.show()


def plot_heatmap(lda_model, corpus, num_docs=10):
    """
    绘制文档-主题概率分布热力图
    """
    # 获取文档主题分布
    doc_topics = []
    for doc in corpus[:num_docs]:
        topics = lda_model.get_document_topics(doc, minimum_probability=0)
        doc_topics.append([prob for topic_id, prob in topics])

    # 创建DataFrame
    df = pd.DataFrame(
        doc_topics,
        columns=[f"Topic {i + 1}" for i in range(lda_model.num_topics)],
        index=[f"Doc {i + 1}" for i in range(len(doc_topics))]
    )

    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Document-Topic Probability Distribution")
    plt.show()


async def analyze_topic_with_deepseek(topic_keywords: List[Tuple[str, float]], topic_id: int) -> str:
    """
    使用DeepSeek R1:8B模型分析主题内容
    """
    # 准备关键词字符串
    keywords_str = ", ".join([word for word, _ in topic_keywords])

    # 构造提示词
    prompt = f"""
    你是一个专业的文本分析专家。请根据以下关键词分析这个主题可能涉及的内容:

    关键词: {keywords_str}

    请按照以下格式提供分析:
    1. 主题概括: 用一句话总结这个主题的核心内容
    2. 可能涉及的内容: 列出3-5个这个主题可能涉及的具体方面
    3. 相关领域: 指出这个主题可能属于哪些领域或学科
    4. 潜在关联: 指出这个主题可能与其他哪些主题相关联

    分析时请注意:
    - 保持专业但易懂
    - 提供具体而非泛泛的分析
    - 考虑关键词之间的关联性
    """

    try:
        # 调用Ollama API
        response = await ollama.AsyncClient().chat(
            model='deepseek-r1:8b',
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': 0.7,
                'top_p': 0.9
            }
        )

        return response['message']['content']

    except Exception as e:
        print(f"调用DeepSeek模型出错: {e}")
        return f"无法获取主题{topic_id}的分析结果"


async def analyze_topics_with_llm(lda_model, num_topics: int):
    """
    使用DeepSeek R1:8B模型分析所有主题内容
    """
    print("\n=== 主题内容深度分析(使用DeepSeek R1:8B) ===")

    tasks = []
    for topic_id in range(num_topics):
        # 获取主题的前10个关键词
        keywords = lda_model.show_topic(topic_id, topn=10)
        tasks.append(analyze_topic_with_deepseek(keywords, topic_id + 1))

    # 并行获取所有主题的分析结果
    analyses = await asyncio.gather(*tasks)

    # 打印分析结果
    for i, analysis in enumerate(analyses, 1):
        print(f"\n◆ 主题 {i} 深度分析 ◆")
        print(analysis)
        print("-" * 80)


# 修改主函数为异步
async def main():
    # 1. 数据准备
    data_file = Path("D:/papers/data/twitter_dataset/devset/posts.txt")
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return

    texts = load_and_sample_data(data_file, sample_size=10)
    if not texts:
        print("无法加载文本数据")
        return

    print("示例文本:")
    for i, text in enumerate(texts[:2], 1):
        print(f"{i}. {text[:100]}...")

    # 2. 数据预处理
    processed_texts = preprocess_text(texts)
    print("\n预处理后的示例文本:")
    print(processed_texts[0][:10], "...")

    # 3. 模型构建与训练
    num_topics = min(3, len(processed_texts))
    lda_model, corpus, dictionary = train_lda_model(processed_texts, num_topics=num_topics)

    # 打印主题
    print("\n发现的主题:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"主题 {idx + 1}: {topic}")

    # 4. 可视化分析
    print("\n生成可视化...")

    # pyLDAvis交互图
    print("\n1. pyLDAvis交互图:")
    vis = visualize_lda(lda_model, corpus, dictionary)

    # 词云图
    print("\n2. 主题词云图:")
    generate_wordclouds(lda_model, num_topics)

    # 热力图
    print("\n3. 文档-主题分布热力图:")
    plot_heatmap(lda_model, corpus)

    # 使用DeepSeek R1:8B模型分析主题内容
    #await analyze_topics_with_llm(lda_model, num_topics)


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())