from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
import json


class BaseAgent(ABC):
    """智能体基类"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = []

    def remember(self, message: str, agent: str = "system"):
        """记忆对话历史"""
        self.memory.append({"agent": agent, "message": message})

    def get_recent_memory(self, n: int = 5) -> List[Dict]:
        """获取最近的记忆"""
        return self.memory[-n:] if self.memory else []

    @abstractmethod
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理查询的抽象方法"""
        pass

    def format_response(self, content: str, task_type: str = "response") -> Dict[str, Any]:
        """格式化响应"""
        return {
            "agent": self.name,
            "content": content,
            "task_type": task_type,
            "timestamp": str(pd.Timestamp.now())
        }