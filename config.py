import os
from langchain_community.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import numpy as np
import requests
from langchain_core.embeddings import Embeddings


class Config:
    # 模型配置
    SILICONFLOW_API_KEY = "sk-aiijdfbzalmwidpetzrzopatkbeotqaxsnuixggvmcvxutcd"
    EMBED_MODEL = "BAAI/bge-m3"
    LLM_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

    # 向量数据库配置
    FAISS_INDEX_PATH = "././faiss_renewed_index"
    BOOKS_DATA_PATH = "./book_embeddings_renewed.csv"  # 书籍数据文件

    # 知识库路径
    KNOWLEDGE_BASE_PATH = "./knowledge_docs"


class SiliconFlowEmbeddings(Embeddings):
    """硅基流动嵌入模型 - 修复版"""

    def __init__(self, model_name=Config.EMBED_MODEL, api_key=Config.SILICONFLOW_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.dimension = 1024

    def embed_query(self, text):
        """为查询生成嵌入向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "input": [text],
            "encoding_format": "float"
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            print(f"❌ 查询向量生成失败: {e}")
            return np.random.normal(0, 0.1, self.dimension).tolist()

    def embed_documents(self, texts):
        """为文档生成嵌入向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 分批处理，避免请求过大
        batch_size = 10
        all_embeddings = []

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  生成文档嵌入批次 {batch_num}/{total_batches}")

            data = {
                "model": self.model_name,
                "input": batch_texts,
                "encoding_format": "float"
            }

            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"  ❌ 文档嵌入批次失败: {e}")
                # 为失败的批次生成随机向量
                all_embeddings.extend([np.random.normal(0, 0.1, self.dimension).tolist() for _ in batch_texts])

            # 避免API限制
            if batch_num < total_batches:
                import time
                time.sleep(1)

        return all_embeddings


class LibraryTools:
    """图书馆智能体可用的工具集"""

    def __init__(self):
        self.vectorstore = None
        self.embeddings = SiliconFlowEmbeddings()
        self.init_tools()

    def init_tools(self):
        """初始化向量数据库"""
        # 如果FAISS索引不存在，创建它
        if not os.path.exists(Config.FAISS_INDEX_PATH):
            self._create_books_vectorstore()
        else:
            self.vectorstore = FAISS.load_local(
                Config.FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"📂 加载FAISS书籍索引成功")

    # config.py 中的 _create_books_vectorstore 方法替换为：

    def _create_books_vectorstore(self):
        """创建基于书籍数据的FAISS向量数据库"""
        import pandas as pd

        # 检查书籍数据文件是否存在
        if not os.path.exists(Config.BOOKS_DATA_PATH):
            print(f"❌ 书籍数据文件不存在: {Config.BOOKS_DATA_PATH}")
            # 创建空的向量存储
            self.vectorstore = FAISS.from_texts(
                texts=["暂无书籍数据"],
                embedding=self.embeddings
            )
            self.vectorstore.save_local(Config.FAISS_INDEX_PATH)
            print(f"💾 创建空FAISS索引: {Config.FAISS_INDEX_PATH}")
            return

        try:
            # 读取书籍数据
            df = pd.read_csv(Config.BOOKS_DATA_PATH, encoding="utf-8-sig")
            print(f"📖 读取到 {len(df)} 条书籍数据")

            # 准备数据
            texts = []
            metadatas = []

            success_count = 0
            for idx, row in df.iterrows():
                try:
                    # 检查必要字段
                    if "text" not in row or "embedding" not in row:
                        continue

                    text = str(row["text"]).strip()
                    if not text:
                        continue

                    # 解析现有的嵌入向量（避免重新生成）
                    embedding_str = str(row["embedding"]).strip()
                    try:
                        embedding = list(map(float, embedding_str.split(",")))
                        if len(embedding) != 1024:
                            continue
                    except:
                        continue

                    texts.append(text)
                    metadatas.append({
                        "title": str(row.get("title", "无题名")),
                        "author": str(row.get("author", "未知作者")),
                        "publisher": str(row.get("publisher", "未知出版社")),
                        "year": str(row.get("year", "未知年份")),
                        "chunk_id": str(row.get("chunk_id", str(idx))),
                        "book_id": str(row.get("book_id", str(idx)))
                    })
                    success_count += 1

                    # 限制数据量用于测试
                    if success_count >= 10000:  # 最多1万条用于测试
                        break

                except Exception as e:
                    continue

            print(f"✅ 准备 {success_count} 条有效记录")

            if success_count == 0:
                raise Exception("没有有效的书籍数据")

            # 使用现有的嵌入向量创建FAISS索引
            print("🔄 使用现有嵌入向量创建FAISS索引...")

            # 提取嵌入向量
            embeddings_list = []
            for idx, row in df.head(success_count).iterrows():
                try:
                    embedding_str = str(row["embedding"]).strip()
                    embedding = list(map(float, embedding_str.split(",")))
                    if len(embedding) == 1024:
                        embeddings_list.append(embedding)
                    else:
                        # 如果嵌入向量无效，使用零向量
                        embeddings_list.append([0.0] * 1024)
                except:
                    embeddings_list.append([0.0] * 1024)

            # 创建FAISS索引
            import numpy as np
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings_list)),
                embedding=self.embeddings,
                metadatas=metadatas
            )

            # 保存索引
            self.vectorstore.save_local(Config.FAISS_INDEX_PATH)
            print(f"💾 书籍FAISS索引已保存到: {Config.FAISS_INDEX_PATH}")
            print(f"📚 索引包含: {success_count} 本书籍")

        except Exception as e:
            print(f"❌ 创建书籍向量库失败: {e}")
            # 创建空的向量存储作为降级方案
            self.vectorstore = FAISS.from_texts(
                texts=["书籍数据库初始化失败"],
                embedding=self.embeddings
            )
            self.vectorstore.save_local(Config.FAISS_INDEX_PATH)

    # 替换 config.py 中的 search_knowledge_base 方法：

    def search_knowledge_base(self, query: str) -> str:
        """搜索知识库工具 - 基于书籍数据"""
        if self.vectorstore is None:
            return "书籍数据库尚未初始化"

        try:
            print(f"🔍 搜索查询: '{query}'")
            docs = self.vectorstore.similarity_search(query, k=10)  # 增加检索数量
            print(f"📄 找到 {len(docs)} 个相关文档")

            if not docs:
                return "未找到相关书籍信息"

            results = []
            seen_books = set()

            for i, doc in enumerate(docs):
                title = doc.metadata.get('title', '无题名')
                author = doc.metadata.get('author', '未知作者')
                book_key = f"{title}-{author}"

                # 去重
                if book_key in seen_books:
                    continue
                seen_books.add(book_key)

                publisher = doc.metadata.get('publisher', '未知出版社')
                year = doc.metadata.get('year', '未知年份')

                book_info = f"《{title}》\n   作者: {author}"
                if publisher != '未知出版社':
                    book_info += f"\n   出版社: {publisher}"
                if year != '未知年份':
                    book_info += f"\n   出版年: {year}"

                # 添加内容预览
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                book_info += f"\n   简介: {content_preview}"

                results.append(book_info)

                # 最多返回8本书
                if len(results) >= 8:
                    break

            return "\n\n".join(results) if results else "未找到相关书籍"

        except Exception as e:
            print(f"❌ 搜索错误: {e}")
            return f"搜索过程中出错: {str(e)}"

    def search_book_catalog(self, query: str) -> str:
        """图书目录搜索工具 - 增强版"""
        if self.vectorstore is None:
            return "书籍数据库尚未初始化"

        try:
            # 使用向量搜索找到相关书籍
            docs = self.vectorstore.similarity_search(query, k=8)
            if not docs:
                return "未找到相关图书"

            # 按作者和类别分组
            author_books = {}
            category_books = {}

            for doc in docs:
                title = doc.metadata.get('title', '无题名')
                author = doc.metadata.get('author', '未知作者')
                publisher = doc.metadata.get('publisher', '未知出版社')
                year = doc.metadata.get('year', '未知年份')

                # 按作者分组
                if author not in author_books:
                    author_books[author] = []
                author_books[author].append(f"《{title}》({year})")

                # 简单分类（根据查询关键词）
                if "小说" in query or "文学" in query:
                    category = "文学小说"
                elif "历史" in query:
                    category = "历史"
                elif "科学" in query or "技术" in query:
                    category = "科学技术"
                else:
                    category = "其他"

                if category not in category_books:
                    category_books[category] = []
                category_books[category].append(f"《{title}》 - {author}")

            # 构建结果
            results = []

            if author_books:
                results.append("按作者分类:")
                for author, books in list(author_books.items())[:3]:  # 最多3个作者
                    results.append(f"  {author}: {', '.join(books[:3])}")

            if category_books:
                results.append("\n按类别分类:")
                for category, books in category_books.items():
                    results.append(f"  {category}: {', '.join(books[:3])}")

            return "\n".join(results) if results else "未找到相关图书"

        except Exception as e:
            return f"目录搜索过程中出错: {str(e)}"

    def get_tools(self):
        """返回所有工具"""
        return [
            Tool(
                name="knowledge_base_search",
                func=self.search_knowledge_base,
                description="用于在图书馆知识库中搜索书籍相关信息，包括书名、作者、出版社等"
            ),
            Tool(
                name="book_catalog_search",
                func=self.search_book_catalog,
                description="用于在图书目录中搜索书籍，提供按作者和分类的搜索结果"
            )
        ]