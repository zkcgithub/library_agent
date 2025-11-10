from orchestrator import MultiAgentOrchestrator
import time


def main():
    """演示主函数"""
    print("图书馆智能问答系统 - FAISS版本")
    print("=" * 50)
    print("系统初始化中...")

    # 初始化协调器
    orchestrator = MultiAgentOrchestrator()

    # 测试用例 - 基于书籍数据的查询
    test_queries = [
        "推荐几本巴金的小说",
        "鲁迅的作品有哪些？",
        "找一些历史类的书籍"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} 测试用例 {i} {'='*20}")
        print(f"📖 用户查询: {query}")

        start_time = time.time()
        result = orchestrator.process_user_query(query)
        end_time = time.time()

        print(f"\n💡 智能回答:")
        print(f"{result['final_answer']}")
        print(f"\n⏱️ 处理时间: {end_time - start_time:.2f}秒")
        print(f"📊 处理步骤: {result['conversation_steps']}步")

        # 显示搜索到的书籍
        if 'task_results' in result:
            print(f"\n🔍 搜索到的书籍:")
            for task in result['task_results']:
                if 'knowledge_base_search' in task.get('result', ''):
                    print(f"  - {task['description']}")


if __name__ == "__main__":
    main()