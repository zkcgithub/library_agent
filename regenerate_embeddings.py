# regenerate_embeddings.py
import os
import pandas as pd
import numpy as np
import requests
import time
from config import Config, SiliconFlowEmbeddings


def regenerate_book_embeddings():
    """重新生成书籍嵌入向量"""
    print("🔄 开始重新生成书籍嵌入向量...")

    # 读取原始数据
    try:
        df = pd.read_csv("book_embeddings.csv", encoding="utf-8-sig")
        print(f"📖 读取到 {len(df)} 条原始数据")
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    # 初始化嵌入生成器
    embeddings = SiliconFlowEmbeddings()

    # 分批重新生成嵌入向量
    BATCH_SIZE = 20  # 减小批次大小避免API限制
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    new_embeddings = []
    success_count = 0
    failed_count = 0

    print(f"🔄 分 {total_batches} 批重新生成嵌入向量...")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(df))

        batch_texts = []
        batch_indices = []

        # 准备批次数据
        for i in range(start_idx, end_idx):
            try:
                text = str(df.iloc[i]["text"]).strip()
                if text and len(text) > 10:  # 确保文本有效
                    batch_texts.append(text)
                    batch_indices.append(i)
            except:
                continue

        if not batch_texts:
            continue

        print(f"处理批次 {batch_idx + 1}/{total_batches} ({len(batch_texts)}条文本)...")

        # 生成嵌入向量
        try:
            batch_embeddings = embeddings.embed_documents(batch_texts)

            for idx, embedding in zip(batch_indices, batch_embeddings):
                if len(embedding) == 1024:
                    # 将嵌入向量转换为字符串存储
                    embedding_str = ",".join(map(str, embedding))
                    new_embeddings.append((idx, embedding_str))
                    success_count += 1
                else:
                    failed_count += 1

            print(f"✅ 批次 {batch_idx + 1} 完成")

        except Exception as e:
            print(f"❌ 批次 {batch_idx + 1} 失败: {e}")
            failed_count += len(batch_texts)

        # 避免API限制
        time.sleep(2)

    print(f"\n📊 重新生成完成:")
    print(f"   - 成功: {success_count}条")
    print(f"   - 失败: {failed_count}条")

    # 更新DataFrame
    if success_count > 0:
        # 创建新的嵌入向量列
        embedding_dict = {idx: emb for idx, emb in new_embeddings}
        df['new_embedding'] = df.index.map(embedding_dict)

        # 保存新的数据文件
        output_file = "book_embeddings_renewed.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"💾 新的嵌入文件已保存: {output_file}")

        return output_file
    else:
        print("❌ 没有成功生成任何嵌入向量")
        return None


def test_new_embeddings(file_path):
    """测试新生成的嵌入向量"""
    print(f"\n🧪 测试新嵌入向量文件: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"📊 数据统计: {len(df)} 条记录")

        # 检查嵌入向量质量
        if 'new_embedding' in df.columns:
            sample_embedding = df.iloc[0]['new_embedding']
            if pd.notna(sample_embedding):
                vector = list(map(float, str(sample_embedding).split(",")))
                print(f"🔍 新嵌入向量检查:")
                print(f"   - 维度: {len(vector)}")
                print(f"   - 范围: [{min(vector):.4f}, {max(vector):.4f}]")
                print(f"   - 均值: {np.mean(vector):.4f}")

        # 检查作者分布
        if 'author' in df.columns:
            author_counts = df['author'].value_counts()
            print(f"📚 作者分布 (前5):")
            for author, count in author_counts.head(5).items():
                print(f"   - {author}: {count}条")

    except Exception as e:
        print(f"❌ 测试失败: {e}")


def create_faiss_with_new_embeddings(file_path):
    """使用新嵌入向量创建FAISS索引"""
    print(f"\n🔧 使用新嵌入向量创建FAISS索引...")

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

        # 准备数据
        texts = []
        metadatas = []
        embeddings_list = []

        success_count = 0
        for _, row in df.iterrows():
            try:
                # 使用新的嵌入向量
                if 'new_embedding' not in row or pd.isna(row['new_embedding']):
                    continue

                embedding_str = str(row['new_embedding']).strip()
                embedding = list(map(float, embedding_str.split(",")))
                if len(embedding) != 1024:
                    continue

                text = str(row["text"]) if "text" in row else ""
                if not text.strip():
                    continue

                texts.append(text)
                metadatas.append({
                    "title": str(row.get("title", "无题名")),
                    "author": str(row.get("author", "未知作者")),
                    "publisher": str(row.get("publisher", "未知出版社")),
                    "year": str(row.get("year", "未知年份")),
                    "chunk_id": str(row.get("chunk_id", "")),
                    "book_id": str(row.get("book_id", ""))
                })
                embeddings_list.append(embedding)
                success_count += 1

            except Exception as e:
                continue

        print(f"✅ 准备 {success_count} 条有效记录")

        if success_count == 0:
            raise Exception("没有有效的记录")

        # 创建FAISS索引
        embeddings = SiliconFlowEmbeddings()
        from langchain_community.vectorstores import FAISS
        import numpy as np

        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_list)),
            embedding=embeddings,
            metadatas=metadatas
        )

        # 保存索引
        new_index_path = "./faiss_renewed_index"
        vectorstore.save_local(new_index_path)
        print(f"💾 新FAISS索引已保存到: {new_index_path}")

        return new_index_path, vectorstore

    except Exception as e:
        print(f"❌ 创建FAISS索引失败: {e}")
        return None, None


def test_search_accuracy(vectorstore):
    """测试搜索准确性"""
    print(f"\n🎯 测试搜索准确性...")

    test_queries = [
        "巴金",
        "鲁迅",
        "小说",
        "历史",
        "老舍",
        "郭沫若"
    ]

    for query in test_queries:
        print(f"\n🔍 搜索: '{query}'")
        try:
            docs = vectorstore.similarity_search(query, k=3)

            for i, doc in enumerate(docs):
                title = doc.metadata.get('title', '无题名')
                author = doc.metadata.get('author', '未知作者')
                print(f"  {i + 1}. 《{title}》 - {author}")

                # 检查是否相关
                if query in author or query in title:
                    print(f"     ✅ 相关!")
                else:
                    print(f"     ❌ 不相关")

        except Exception as e:
            print(f"  ❌ 搜索失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("🔄 书籍嵌入向量重新生成工具")
    print("=" * 60)

    # 步骤1: 重新生成嵌入向量
    print("\n1. 重新生成嵌入向量")
    new_file = regenerate_book_embeddings()

    if not new_file:
        print("❌ 重新生成失败，退出程序")
        return

    # 步骤2: 测试新嵌入向量
    print("\n2. 测试新嵌入向量")
    test_new_embeddings(new_file)

    # 步骤3: 创建新的FAISS索引
    print("\n3. 创建新的FAISS索引")
    new_index_path, vectorstore = create_faiss_with_new_embeddings(new_file)

    if vectorstore:
        # 步骤4: 测试搜索准确性
        print("\n4. 测试搜索准确性")
        test_search_accuracy(vectorstore)

        print(f"\n🎉 重新生成完成!")
        print(f"   新数据文件: {new_file}")
        print(f"   新索引路径: {new_index_path}")
        print(f"\n💡 请更新 config.py 中的路径配置:")
        print(f"   FAISS_INDEX_PATH = '{new_index_path}'")
        print(f"   BOOKS_DATA_PATH = '{new_file}'")


if __name__ == "__main__":
    main()