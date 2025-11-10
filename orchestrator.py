import time
from user_agent import UserAgent  # 确保导入修复后的UserAgent
from library_agent import LibraryAgent


class MultiAgentOrchestrator:
    """多智能体协调器 - 完整版"""

    def __init__(self):
        self.conversation_history = []
        self.user_agent = UserAgent()
        self.library_agent = LibraryAgent()
        print("✅ 多智能体系统初始化完成")

    def process_user_query(self, query: str) -> dict:
        """处理用户查询"""
        print(f"\n=== 开始处理用户查询 ===")
        print(f"用户查询: {query}")

        steps = 0
        start_time = time.time()

        try:
            # 步骤1: 用户智能体分析意图和规划任务
            steps += 1
            print("--- 用户智能体规划任务 ---")
            user_response = self.user_agent.process_query(query)

            if "tasks" not in user_response or not user_response["tasks"]:
                return {
                    "final_answer": "抱歉，我没有理解您的需求。请尝试更具体地描述您想找什么书籍。",
                    "conversation_steps": steps,
                    "processing_time": time.time() - start_time,
                    "task_results": []
                }

            print(f"规划任务: {len(user_response['tasks'])}个")

            # 步骤2: 图书馆智能体执行任务
            steps += 1
            print("--- 图书馆智能体执行任务 ---")
            library_response = self.library_agent.process_query(user_response)

            # 步骤3: 生成最终回答
            steps += 1
            if "summary" in library_response:
                final_answer = library_response["summary"]
            elif "response" in library_response:
                final_answer = library_response["response"]
            else:
                final_answer = "已为您搜索相关信息。"

            print("=== 查询处理完成 ===")

            return {
                "final_answer": final_answer,
                "conversation_steps": steps,
                "processing_time": time.time() - start_time,
                "task_results": library_response.get("task_results", []),
                "task_details": library_response.get("task_results", [])
            }

        except Exception as e:
            print(f"❌ 处理过程出错: {e}")
            return {
                "final_answer": f"处理过程中出现错误: {str(e)}",
                "conversation_steps": steps,
                "processing_time": time.time() - start_time,
                "task_results": []
            }