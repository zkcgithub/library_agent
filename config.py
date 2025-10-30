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
        if self.vectorstore is None:
            return "知识库尚未初始化"

        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            results = []
            for i, doc in enumerate(docs):
                results.append(f"文档 {i + 1}:\n{doc.page_content}\n")
            return "\n".join(results) if results else "未找到相关信息"
        except Exception as e:
            return f"搜索过程中出错: {str(e)}"

    def search_book_catalog(self, query: str) -> str:
        """模拟图书目录搜索工具"""
        # 这里可以替换为真实的图书馆API调用
        mock_books = [
            {"title": "深度学习", "author": "Ian Goodfellow", "year": 2017, "category": "计算机科学"},
            {"title": "人工智能：现代方法", "author": "Stuart Russell", "year": 2020, "category": "计算机科学"},
            {"title": "Python编程从入门到实践", "author": "Eric Matthes", "year": 2020, "category": "编程"},
        ]

        # 简单关键词匹配
        results = []
        for book in mock_books:
            if any(keyword.lower() in str(book.values()).lower()
                   for keyword in query.split()):
                results.append(f"{book['title']} - {book['author']} ({book['year']})")

        return "\n".join(results) if results else "未找到相关图书"

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