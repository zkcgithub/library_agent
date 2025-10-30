import streamlit as st
from orchestrator import MultiAgentOrchestrator


def main():
    st.title("🏛️ 图书馆多智能体协作问答系统")
    st.markdown("这是一个基于双智能体协作的图书馆问答系统原型")

    # 初始化协调器
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = MultiAgentOrchestrator()

    # 用户输入
    user_query = st.text_input("请输入您的问题:", placeholder="例如：帮我找深度学习的书籍")

    if st.button("提交查询") and user_query:
        with st.spinner("智能体正在协作处理您的查询..."):
            result = st.session_state.orchestrator.process_user_query(user_query)

            # 显示结果
            st.success("查询处理完成！")

            st.subheader("🤖 最终回答:")
            st.write(result["final_answer"])

            with st.expander("查看任务执行详情"):
                for task in result.get("task_details", []):
                    st.write(f"**任务 {task['task_id']}**: {task['description']}")
                    st.text_area(f"任务 {task['task_id']} 结果",
                                 task['result'],
                                 height=150,
                                 key=f"task_{task['task_id']}")


if __name__ == "__main__":
    main()