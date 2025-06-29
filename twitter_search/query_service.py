import faiss
import json
import numpy as np
from InstructorEmbedding import INSTRUCTOR
import torch


class TwitterSearcher:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = INSTRUCTOR('hkunlp/instructor-large', device=self.device)
        self.index = faiss.read_index("twitter_index.faiss")
        with open("twitter_metadata.json", "r") as f:
            self.metadata = json.load(f)

    def search(self, query: str, k: int = 5):
        # 生成查询向量
        query_embedding = self.model.encode(
            [["Represent the Twitter search query:", query]],
            normalize_embeddings=True
        ).astype('float32')

        # 执行搜索
        distances, indices = self.index.search(query_embedding, k)

        # 包装结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            post = self.metadata[idx]
            results.append({
                "post_id": post["post_id"],
                "text": post["text"],
                "user_id": post["user_id"],
                "timestamp": post["timestamp"],
                "score": float(dist)  # 相似度分数
            })
        return results


# 使用示例
if __name__ == "__main__":
    searcher = TwitterSearcher()

    while True:
        query = input("\n输入搜索查询 (或输入 'quit' 退出): ")
        if query.lower() == 'quit':
            break

        results = searcher.search(query, k=3)

        print(f"\n找到 {len(results)} 条相关推文:")
        for i, result in enumerate(results, 1):
            print(f"\n结果 #{i} (相似度: {result['score']:.3f})")
            print(f"用户: @{result['user_id']}")
            print(f"时间: {result['timestamp']}")
            print(f"内容: {result['text']}")
            print("-" * 80)