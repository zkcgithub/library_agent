from base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from config import Config
import json


class UserAgent(BaseAgent):
    """用户智能体 - 负责理解用户意图和规划任务"""

    def __init__(self):
        super().__init__("UserAgent", "用户意图理解与任务规划")
        self.llm = ChatOpenAI(
            api_key=Config.SILICONFLOW_API_KEY,
            base_url=Config.SILICONFLOW_API_BASE,
            model=Config.LLM_MODEL,
            temperature=0.1,
            max_tokens=800
        )

    def understand_intent(self, query: str) -> dict:
        """理解用户意图"""
        prompt = f"""
请分析以下用户查询的意图，并确定需要执行的任务：

用户查询: "{query}"

请按以下JSON格式返回分析结果：
{{
    "intent": "搜索|推荐|咨询|其他",
    "target_type": "书籍|作者|主题|其他", 
    "target_details": "具体的目标描述",
    "required_tools": ["knowledge_base_search", "book_catalog_search"],
    "tasks": [
        {{
            "type": "search",
            "description": "具体的任务描述",
            "tools": ["knowledge_base_search", "book_catalog_search"]
        }}
    ]
}}

请确保tasks数组至少包含一个任务。
"""

        try:
            response = self.llm.invoke(prompt)
            return json.loads(response.content)
        except:
            # 如果LLM解析失败，使用基于规则的回退
            return self._fallback_intent_understanding(query)

    def _fallback_intent_understanding(self, query: str) -> dict:
        """基于规则的回退意图理解"""
        tasks = [{
            "type": "search",
            "description": f"搜索关于'{query}'的书籍信息",
            "tools": ["knowledge_base_search", "book_catalog_search"]
        }]

        return {
            "intent": "搜索",
            "target_type": "书籍",
            "target_details": query,
            "required_tools": ["knowledge_base_search", "book_catalog_search"],
            "tasks": tasks
        }

    def plan_tasks(self, query: str) -> dict:
        """规划执行任务"""
        intent_analysis = self.understand_intent(query)

        response = self.format_response("任务规划完成", "task_plan")
        response.update({
            "original_query": query,
            "intent_analysis": intent_analysis,
            "tasks": intent_analysis['tasks'],
            "next_agent": "LibraryAgent"
        })

        return response

    def process_query(self, query: str, context: dict = None) -> dict:
        """处理用户查询"""
        self.remember(f"处理用户查询: {query}")
        return self.plan_tasks(query)