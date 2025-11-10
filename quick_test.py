# quick_test.py
import pandas as pd
from config import SiliconFlowEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np


def create_small_test_set():
    """创建小型测试集快速验证"""
    print("🔬 创建小型测试集...")

    try:
        # 读取原始数据
        df = pd.read_csv("book_embeddings.csv", encoding="utf-8-sig")

        # 选择包含知名作者的小样本
        target_authors = ["巴金", "鲁迅", "梁启超", "郭沫若", "老舍"]
        test_df = df[df['author'].isin(target_authors)].head(50)

        if len(test_df) == 0:
            print("❌ 未找到目标作者的记录，使用随机样本")
            test_df = df.head(50)

        print(f"✅ 创建测试集: {len(test_df)} 条记录")

        # 重新生成嵌入向量
        embeddings = SiliconFlowEmbeddings()
        texts = test_df['text'].astype(str).tolist()

        print("🔄 重新生成嵌入向量...")
        new_embeddings = embeddings.embed_documents(texts)

        # 准备数据
        texts_clean = []
        metadatas = []
        embeddings_list = []

        for i, (idx, row) in enumerate(test_df.iterrows()):
            if i < len(new_embeddings) and len(new_embeddings[i]) == 1024:
                texts_clean.append(str(row["text"]))
                metadatas.append({
                    "title": str(row.get("title", "无题名")),
                    "author": str(row.get("author", "未知作者")),
                    "publisher": str(row.get("publisher", "未知出版社")),
                    "year": str(row.get("year", "未知年份"))
                })
                embeddings_list.append(new_embeddings[i])

        print(f"✅ 准备 {len(texts_clean)} 条有效记录")

        # 创建FAISS索引
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts_clean, embeddings_list)),
            embedding=embeddings,
            metadatas=metadatas
        )

        # 测试搜索
        print("\n🎯 测试搜索准确性:")
        test_queries = ["巴金", "鲁迅", "小说"]

        for query in test_queries:
            print(f"\n🔍 搜索: '{query}'")
            docs = vectorstore.similarity_search(query, k=3)

            for i, doc in enumerate(docs):
                title = doc.metadata.get('title', '无题名')
                author = doc.metadata.get('author', '未知作者')
                print(f"  {i + 1}. 《{title}》 - {author}")

                if query in author:
                    print(f"     ✅ 相关!")

        return vectorstore

    except Exception as e:
        print(f"❌ 创建测试集失败: {e}")
        return None


if __name__ == "__main__":
    create_small_test_set()