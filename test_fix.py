from config import LibraryTools


def test_vector_db():
    """测试向量数据库"""
    print("🧪 测试向量数据库...")

    tools = LibraryTools()

    # 测试搜索
    test_queries = [
        "巴金",
        "鲁迅",
        "小说",
        "历史"
    ]

    for query in test_queries:
        print(f"\n🔍 测试搜索: '{query}'")
        result = tools.search_knowledge_base(query)
        print(f"结果: {result[:200]}...")


def test_specific_books():
    """测试特定书籍搜索"""
    print("\n🎯 测试特定书籍搜索...")

    tools = LibraryTools()

    # 直接测试巴金的作品
    print("直接搜索巴金作品:")
    docs = tools.vectorstore.similarity_search("巴金", k=5)

    for i, doc in enumerate(docs):
        title = doc.metadata.get('title', '无题名')
        author = doc.metadata.get('author', '未知作者')
        print(f"{i + 1}. 《{title}》 - {author}")


if __name__ == "__main__":
    test_vector_db()
    test_specific_books()