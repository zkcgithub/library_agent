from user_agent import UserAgent
from library_agent import LibraryAgent


class MultiAgentOrchestrator:
    """多智能体协作协调器"""

    def __init__(self):
        self.user_agent = UserAgent()
        self.library_agent = LibraryAgent()
        self.conversation_history = []

    def process_user_query(self, user_query: str) -> dict[str, any]:
        """处理用户查询的完整流程"""
        print(f"\n=== 开始处理用户查询 ===")
        print(f"用户查询: {user_query}")

        # 1. 用户智能体分析查询并规划任务
        print("\n--- 用户智能体规划任务 ---")
        plan_response = self.user_agent.process_query(user_query)
        self.conversation_history.append(plan_response)
        print(f"规划任务: {len(plan_response['tasks'])}个")

        # 2. 图书馆智能体执行任务
        print("\n--- 图书馆智能体执行任务 ---")
        execution_response = self.library_agent.process_query(plan_response)
        self.conversation_history.append(execution_response)
        print("任务执行完成")

        # 3. 返回最终结果
        final_response = {
            "original_query": user_query,
            "final_answer": execution_response["content"],
            "task_details": execution_response.get("task_results", []),
            "conversation_steps": len(self.conversation_history)
        }

        print(f"\n=== 查询处理完成 ===")
        return final_response

    def get_conversation_history(self):
        """获取对话历史"""
        return self.conversation_history