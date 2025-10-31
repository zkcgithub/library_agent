from orchestrator import MultiAgentOrchestrator


def main():
    """演示主函数"""
    print("图书馆多智能体协作系统原型演示")
    print("=" * 50)

    # 初始化协调器
    orchestrator = MultiAgentOrchestrator()

    # 测试用例
    test_queries = [
        "帮我找深度学习的书籍",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 20} 测试用例 {i} {'=' * 20}")
        print(f"查询: {query}")

        result = orchestrator.process_user_query(query)

        print(f"\n最终回答:")
        print(f"{result['final_answer']}")
        print(f"\n处理步骤: {result['conversation_steps']}步")


if __name__ == "__main__":
    main()