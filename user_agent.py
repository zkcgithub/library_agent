import re
from base_agent import BaseAgent


class UserAgent(BaseAgent):
    """用户智能体 - 负责任务规划和分解"""

    def __init__(self):
        super().__init__("UserAgent", "任务规划与分解")
        self.task_patterns = {
            "search": ["查找", "搜索", "找", "查询", "检索"],
            "recommend": ["推荐", "有什么好", "适合"],
            "explain": ["解释", "什么是", "介绍", "说明"],
            "compare": ["对比", "比较", "区别"]
        }

    def analyze_intent(self, query: str) -> dict[str, any]:
        """分析用户意图"""
        intent = {
            "type": [],
            "keywords": [],
            "complexity": "simple"
        }

        # 提取意图类型
        for intent_type, patterns in self.task_patterns.items():
            if any(pattern in query for pattern in patterns):
                intent["type"].append(intent_type)

        # 提取关键词 (简单实现)
        words = query.replace("?", "").replace("？", "").split()
        intent["keywords"] = [word for word in words if len(word) > 1]

        # 判断复杂度
        if len(intent["type"]) > 1 or "compare" in intent["type"]:
            intent["complexity"] = "complex"

        if not intent["type"]:
            intent["type"] = ["search"]  # 默认搜索意图

        return intent

    def plan_tasks(self, intent: dict[str, any]) -> list[dict]:
        """根据意图规划任务"""
        tasks = []

        if "search" in intent["type"]:
            tasks.append({
                "type": "search",
                "description": f"搜索关于{''.join(intent['keywords'][:3])}的相关信息",
                "tools": ["book_catalog_search", "knowledge_base_search"]
            })

        if "explain" in intent["type"]:
            tasks.append({
                "type": "explain",
                "description": f"解释{''.join(intent['keywords'][:2])}的相关概念",
                "tools": ["knowledge_base_search"]
            })

        if "compare" in intent["type"]:
            # 复杂任务分解
            tasks.extend([
                {
                    "type": "search",
                    "description": f"搜索{intent['keywords'][0]}的信息",
                    "tools": ["knowledge_base_search"]
                },
                {
                    "type": "search",
                    "description": f"搜索{intent['keywords'][1] if len(intent['keywords']) > 1 else '相关对比项'}的信息",
                    "tools": ["knowledge_base_search"]
                },
                {
                    "type": "compare",
                    "description": "对比分析找到的信息",
                    "tools": []
                }
            ])

        return tasks

    def process_query(self, query: str, context: dict[str, any] = None) -> dict[str, any]:
        """处理用户查询"""
        self.remember(f"用户查询: {query}", "user")

        # 分析意图
        intent = self.analyze_intent(query)
        self.remember(f"分析意图: {intent}")

        # 规划任务
        tasks = self.plan_tasks(intent)
        self.remember(f"生成任务: {len(tasks)}个任务")

        response = self.format_response("", "task_plan")
        response.update({
            "original_query": query,
            "intent": intent,
            "tasks": tasks,
            "next_agent": "LibraryAgent"
        })

        return response