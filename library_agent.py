from base_agent import BaseAgent
from config import LibraryTools, Config
from langchain_openai import ChatOpenAI


class LibraryAgent(BaseAgent):
    """图书馆智能体 - 负责执行具体任务"""

    def __init__(self):
        super().__init__("LibraryAgent", "书籍检索与推荐")
        self.tools_manager = LibraryTools()
        self.available_tools = {tool.name: tool for tool in self.tools_manager.get_tools()}

        # 初始化LLM - 使用硅基流动API
        self.llm = ChatOpenAI(
            api_key=Config.SILICONFLOW_API_KEY,
            base_url=Config.SILICONFLOW_API_BASE,
            model=Config.LLM_MODEL,
            temperature=0.1,
            max_tokens=1000
        )

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
                        # 从描述中提取查询词，支持中文关键词
                        query_keywords = self._extract_search_query(description)
                        result = self.available_tools[tool_name].func(query_keywords)
                        results.append(f"【{tool_name} 搜索结果】\n{result}")
                    except Exception as e:
                        results.append(f"工具 {tool_name} 执行出错: {str(e)}")

            # 如果是搜索类型的任务，使用LLM进行总结和推荐
            if task_type in ["search", "recommend"] and results:
                prompt = self._build_summary_prompt(description, results)
                llm_result = self.llm.invoke(prompt)
                results.append(f"【智能总结与推荐】\n{llm_result.content}")

            return "\n\n".join(results)
        else:
            # 无工具任务，使用LLM处理
            prompt = f"请处理以下图书馆相关任务：{description}"
            llm_result = self.llm.invoke(prompt)
            return f"任务处理结果:\n{llm_result.content}"

    def _extract_search_query(self, description: str) -> str:
        """从任务描述中提取搜索关键词"""
        # 移除常见的任务描述词汇
        stop_words = ["搜索", "查找", "推荐", "找一下", "关于", "的", "有哪些", "什么"]
        query = description
        for word in stop_words:
            query = query.replace(word, "")
        return query.strip()

    def _build_summary_prompt(self, original_query: str, search_results: list) -> str:
        """构建总结提示词"""
        results_text = "\n".join(search_results)

        prompt = f"""根据用户查询和搜索结果，提供有用的书籍推荐和总结。

用户查询：{original_query}

搜索结果：
{results_text}

请根据以上信息：
1. 总结找到的相关书籍
2. 推荐最相关的3-5本书籍
3. 简要说明推荐理由
4. 如果搜索结果不足，建议用户提供更具体的信息

请用中文回复，保持友好和专业的语气。"""

        return prompt

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

        # 提取所有结果中的关键信息
        all_results = []
        for task in task_results:
            all_results.append(task['result'])

        # 使用LLM进行最终总结
        summary_prompt = f"""用户查询：{original_query}

所有搜索结果：
{"\n".join(all_results)}

请根据以上信息提供一个简洁、有用的最终回答，包括：
1. 主要找到的书籍
2. 关键推荐
3. 下一步建议

用中文回复，保持专业和友好："""

        try:
            llm_result = self.llm.invoke(summary_prompt)
            return llm_result.content
        except Exception as e:
            # 如果LLM总结失败，返回简单汇总
            simple_summary = f"针对您的查询『{original_query}』，我找到了以下信息：\n"
            for task in task_results:
                simple_summary += f"\n{task['description']}:\n"
                # 显示前200字符
                preview = task['result'][:200] + "..." if len(task['result']) > 200 else task['result']
                simple_summary += f"  {preview}\n"
            return simple_summary