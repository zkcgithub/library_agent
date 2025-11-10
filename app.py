import streamlit as st
from orchestrator import MultiAgentOrchestrator
import time
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="智能图书馆问答系统",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_system():
    """初始化系统"""
    if "orchestrator" not in st.session_state:
        with st.spinner("初始化智能体系统..."):
            try:
                st.session_state.orchestrator = MultiAgentOrchestrator()
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"系统初始化失败: {str(e)}")
                st.session_state.initialized = False
    return st.session_state.get("initialized", False)


def display_search_results(result):
    """显示搜索结果"""
    st.subheader("📚 找到的书籍")

    # 从结果中提取书籍信息
    books_found = []
    if "task_results" in result:
        for task in result["task_results"]:
            if "result" in task and "《" in task["result"]:
                # 简单解析书籍信息
                lines = task["result"].split('\n')
                for line in lines:
                    if "《" in line and "》" in line:
                        books_found.append(line.strip())

    if books_found:
        for i, book in enumerate(books_found[:10]):  # 最多显示10本
            st.write(f"{i + 1}. {book}")
    else:
        st.info("未找到具体书籍信息")


def display_processing_details(result):
    """显示处理详情"""
    with st.expander("🔍 查看处理详情", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("处理步骤", result.get("conversation_steps", "N/A"))

        with col2:
            processing_time = result.get("processing_time", 0)
            st.metric("处理时间", f"{processing_time:.2f}秒")

        with col3:
            if "task_results" in result:
                st.metric("执行任务", len(result["task_results"]))

        # 显示任务详情
        if "task_results" in result:
            st.subheader("任务执行详情")
            for task in result["task_results"]:
                with st.expander(f"任务 {task['task_id']}: {task['description']}", expanded=False):
                    st.text_area("", task['result'], height=150, key=f"task_{task['task_id']}")


def main():
    # 侧边栏
    with st.sidebar:
        st.title("🏛️ 智能图书馆系统")
        st.markdown("---")
        st.markdown("### 系统信息")

        # 显示系统状态
        if initialize_system():
            st.success("✅ 系统已就绪")
        else:
            st.error("❌ 系统初始化失败")

        st.markdown("---")
        st.markdown("### 使用说明")
        st.info("""
        您可以询问：
        - 书籍推荐（作者、类型）
        - 书籍搜索
        - 作者作品查询
        - 出版信息等
        """)

        st.markdown("---")
        st.markdown("### 示例问题")
        examples = [
            "推荐几本巴金的小说",
            "鲁迅的作品有哪些？",
            "找一些历史类的书籍",
            "老舍的代表作"
        ]
        for example in examples:
            if st.button(example, key=example):
                st.session_state.user_query = example

    # 主界面
    st.title("🏛️ 智能图书馆问答系统")
    st.markdown("基于多智能体协作的图书馆书籍检索与推荐系统")

    # 用户输入区域
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "💬 请输入您的问题:",
            placeholder="例如：帮我找历史相关的书籍 或 推荐几本巴金的小说",
            value=st.session_state.get("user_query", "")
        )
    with col2:
        st.write("")  # 垂直间距
        submit_btn = st.button("🚀 开始查询", use_container_width=True)

    # 处理查询
    if submit_btn and user_query:
        if not initialize_system():
            st.error("系统未正确初始化，请刷新页面重试")
            return

        # 清空之前的用户查询状态
        if "user_query" in st.session_state:
            del st.session_state.user_query

        # 创建进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 模拟进度更新
            for i in range(3):
                progress_bar.progress((i + 1) * 25)
                status_text.text(f"{['分析意图', '规划任务', '执行搜索', '生成回答'][i]}...")
                time.sleep(0.5)

            # 执行查询
            start_time = time.time()
            result = st.session_state.orchestrator.process_user_query(user_query)
            processing_time = time.time() - start_time

            progress_bar.progress(100)
            status_text.text("查询完成！")
            time.sleep(0.5)

            # 清空进度显示
            progress_bar.empty()
            status_text.empty()

            # 显示结果
            st.success(f"✅ 查询完成 (耗时: {processing_time:.2f}秒)")

            # 主要回答
            st.subheader("💡 智能回答")
            st.write(result["final_answer"])

            # 显示找到的书籍
            display_search_results(result)

            # 处理详情
            display_processing_details(result)

        except Exception as e:
            st.error(f"❌ 查询过程中出现错误: {str(e)}")
            st.info("💡 建议：请尝试重新表述您的问题，或联系系统管理员")

    # 空状态提示
    elif not user_query:
        st.info("💡 请在左侧输入您的问题，或点击示例问题开始查询")

    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        智能图书馆问答系统 | 基于FAISS向量检索 | 多智能体协作
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()