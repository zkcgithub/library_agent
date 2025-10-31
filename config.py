import os
from langchain_community.tools import Tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader


class Config:
    # 模型配置 - 可以使用本地模型或云端API
    LOCAL_LLM = True  # 设为False可使用OpenAI等云端服务

    # 向量数据库配置
    PERSIST_DIRECTORY = "chroma_db"
    EMBEDDING_MODEL = "./all-mpnet-base-v2"

    # 知识库路径
    KNOWLEDGE_BASE_PATH = "./knowledge_docs"

    @classmethod
    def init_embeddings(cls):
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(model_name=cls.EMBEDDING_MODEL)


class LibraryTools:
    """图书馆智能体可用的工具集"""

    def __init__(self):
        self.vectorstore = None
        self.init_tools()

    def init_tools(self):
        """初始化向量数据库"""
        embeddings = Config.init_embeddings()

        # 如果向量数据库不存在，创建它
        if not os.path.exists(Config.PERSIST_DIRECTORY):
            self._create_vectorstore(embeddings)
        else:
            self.vectorstore = Chroma(
                persist_directory=Config.PERSIST_DIRECTORY,
                embedding_function=embeddings
            )

    def _create_vectorstore(self, embeddings):
        """创建向量数据库"""
        documents = []

        # 加载知识库文档
        if os.path.exists(Config.KNOWLEDGE_BASE_PATH):
            for filename in os.listdir(Config.KNOWLEDGE_BASE_PATH):
                file_path = os.path.join(Config.KNOWLEDGE_BASE_PATH, filename)
                try:
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    else:
                        loader = TextLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        if documents:
            # 分割文本
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            # 创建向量存储
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=Config.PERSIST_DIRECTORY
            )
        else:
            # 创建空的向量存储
            self.vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=Config.PERSIST_DIRECTORY
            )

    def search_knowledge_base(self, query: str) -> str:
        """搜索知识库工具"""
        # 改进的知识库搜索工具
        if self.vectorstore is None:
            return "知识库尚未初始化"

        try:
            # 使用更智能的搜索
            docs = self.vectorstore.similarity_search(query, k=5)
            if not docs:
                # 尝试语义相近的搜索
                return "在知识库中未找到相关信息。建议：1. 检查关键词拼写 2. 尝试更通用的术语"

            results = ["📚 知识库检索结果："]
            for i, doc in enumerate(docs, 1):
                content = doc.page_content
                # 简化和格式化输出
                if len(content) > 500:
                    content = content[:500] + "..."
                results.append(f"{i}. {content}")

            return "\n".join(results)
        except Exception as e:
            return f"搜索过程中出错: {str(e)}"

    def search_book_catalog(self, query: str) -> str:
        """模拟图书目录搜索工具"""
        # 这里可以替换为真实的图书馆API调用
        mock_books = [
            {"title": "深度学习", "author": "Ian Goodfellow", "year": 2016,
             "category": "计算机科学", "call_number": "TP181/G646", "status": "可借"},
            {"title": "Python编程从入门到实践", "author": "Eric Matthes", "year": 2016,
             "category": "编程", "call_number": "TP311.56/M429", "status": "可借"},
            {"title": "人工智能：现代方法", "author": "Stuart Russell", "year": 2020,
             "category": "计算机科学", "call_number": "TP18/R961", "status": "可借"},
            {"title": "统计学习方法", "author": "李航", "year": 2019,
             "category": "计算机科学", "call_number": "TP181/L175", "status": "可借"},
            {"title": "机器学习", "author": "周志华", "year": 2016,
             "category": "计算机科学", "call_number": "TP181/Z774", "status": "借出"},
            {"title": "神经网络与深度学习", "author": "Michael Nielsen", "year": 2019,
             "category": "计算机科学", "call_number": "TP183/N669", "status": "可借"},
        ]

        # 改进的搜索逻辑
        results = []
        query_lower = query.lower()

        for book in mock_books:
            # 多字段匹配
            match_score = 0
            if any(keyword in book['title'].lower() for keyword in ['深度学习', '机器学习', '人工智能'] if
                   keyword in query_lower):
                match_score += 2
            if any(keyword in book['title'].lower() for keyword in query_lower.split()):
                match_score += 1
            if any(keyword in book['author'].lower() for keyword in query_lower.split()):
                match_score += 1
            if any(keyword in book['category'] for keyword in ['计算机', '编程', '智能'] if keyword in query_lower):
                match_score += 1

            if match_score > 0:
                results.append((match_score, book))

        # 按匹配度排序
        results.sort(key=lambda x: x[0], reverse=True)

        if results:
            output = ["📖 图书检索结果："]
            for score, book in results[:3]:  # 返回前3个结果
                output.append(
                    f"· 《{book['title']}》 - {book['author']} ({book['year']})\n"
                    f"  类别: {book['category']} | 索书号: {book['call_number']} | 状态: {book['status']}"
                )
            return "\n".join(output)
        else:
            return "未找到相关图书。建议：1. 检查书名或作者名 2. 尝试更通用的搜索词"

    def get_tools(self):
        """返回所有工具"""
        return [
            Tool(
                name="knowledge_base_search",
                func=self.search_knowledge_base,
                description="用于在图书馆知识库中搜索相关信息"
            ),
            Tool(
                name="book_catalog_search",
                func=self.search_book_catalog,
                description="用于在图书目录中搜索书籍"
            )
        ]