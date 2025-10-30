from base_agent import BaseAgent
from config import LibraryTools


class LibraryAgent(BaseAgent):
    """图书馆智能体 - 负责执行具体任务"""

    def __init__(self):
        super().__init__("LibraryAgent", "任务执行与工具调用")
        self.tools_manager = LibraryTools()
        self.available_tools = {tool.name: tool for tool in self.tools_manager.get_tools()}

    def execute_task(self, task: dict) -> str:
        """执行单个任务"""
        task_type = task["type"]
        description = task["description"]
        tools = task.get("tools", [])

        self.remember(f"执行任务: {description}")

        # 根据任务类型选择工具和执行策略
        if tools:
            # 使用工具执行任务
            results = []
            for tool_name in tools:
                if tool_name in self.available_tools:
                    try:
                        # 简单地从描述中提取查询词
                        query_keywords = description.replace("搜索", "").replace("关于", "").replace("的", "")
                        result = self.available_tools[tool_name].func(query_keywords)
                        results.append(f"工具 {tool_name} 结果:\n{result}")
                    except Exception as e:
                        results.append(f"工具 {tool_name} 执行出错: {str(e)}")
            return "\n\n".join(results)
        else:
            # 无工具任务，返回提示信息
            return f"需要人工处理的任务: {description}"

    def process_query(self, query: dict, context: dict[str, any] = None) -> dict[str, any]:
        """处理任务执行请求"""
        if "tasks" not in query:
            return self.format_response("错误: 未找到任务信息")

        self.remember(f"接收任务: {len(query['tasks'])}个任务")

        # 执行所有任务
        task_results = []
        for i, task in enumerate(query["tasks"]):
            self.remember(f"开始执行任务 {i + 1}: {task['description']}")
            result = self.execute_task(task)
            task_results.append({
                "task_id": i + 1,
                "description": task["description"],
                "result": result
            })
            self.remember(f"任务 {i + 1} 完成")

        # 汇总结果
        summary = self.summarize_results(task_results, query.get("original_query", ""))

        response = self.format_response(summary, "task_results")
        response.update({
            "task_results": task_results,
            "summary": summary,
            "next_agent": "UserAgent"  # 返回给用户智能体进行总结
        })

        return response

    def summarize_results(self, task_results: list[dict], original_query: str) -> str:
        """汇总任务结果"""
        if not task_results:
            return "未找到相关信息"

        summary_parts = [f"针对您的查询『{original_query}』，我找到了以下信息：\n"]

        for task in task_results:
            summary_parts.append(f"\n{task['description']}:")
            # 简化结果显示
            result_preview = task['result'][:200] + "..." if len(task['result']) > 200 else task['result']
            summary_parts.append(f"   {result_preview}")

        summary_parts.append("\n以上是初步查找结果，如需更详细信息，请提供更具体的需求。")

        return "\n".join(summary_parts)