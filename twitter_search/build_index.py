import json
import numpy as np
import faiss
from InstructorEmbedding import INSTRUCTOR
from tqdm import tqdm
import torch

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = INSTRUCTOR('hkunlp/instructor-large', device=device)
dimension = 768  # instructor-large的维度
index = faiss.IndexFlatIP(dimension)  # 使用内积相似度

# Twitter数据加载函数
def load_twitter_data(filepath):
    posts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')  # 读取标题行
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:  # 至少需要post_id和post_text
                post = {
                    "post_id": parts[0],
                    "text": parts[1],
                    # 其他字段可按需添加
                    "user_id": parts[2] if len(parts) > 2 else None,
                    "timestamp": parts[5] if len(parts) > 5 else None
                }
                posts.append(post)
    return posts

# 加载Twitter数据
twitter_data = load_twitter_data("E:\codes\pythonPR\twitter_search\data\posts.txt")

# 批量生成嵌入向量
batch_size = 64
texts = [post["text"] for post in twitter_data]
instructions = ["Represent the Twitter post for retrieval:"] * len(texts)

embeddings = []
for i in tqdm(range(0, len(texts), desc="Generating embeddings"):
    batch_texts = texts[i:i+batch_size]
    batch_instructions = instructions[i:i+batch_size]
    batch_embeddings = model.encode(
        [[instr, txt] for instr, txt in zip(batch_instructions, batch_texts)],
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True  # 归一化后可用内积代替余弦相似度
    )
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings).astype('float32')

# 构建FAISS索引
index.add(embeddings)

# 保存索引和数据
faiss.write_index(index, "twitter_index.faiss")
with open("twitter_metadata.json", "w") as f:
    json.dump(twitter_data, f)

print(f"索引构建完成，共处理 {len(twitter_data)} 条推文")