from rag import qa_chain
import gradio as gr

def chat(question):
    if not question or question.strip() == "":
        return "请输入有效的问题～"

    try:
        result = qa_chain.invoke({"question": question.strip()})
        return result["result"]
    except Exception as e:
        return f"回答生成失败：{str(e)}\n请检查：1.Ollama是否启动 2.math.txt是否存在 3.qwen:0.5b模型是否下载"

with gr.Blocks(title="通用本地知识库问答系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 通用本地知识库问答系统
    基于 RAG 检索增强生成 + 本地大模型（qwen:0.5b），回答严谨、无幻觉、不编造
    > 严格根据文档内容回答，不使用外部知识
    """)

    with gr.Row():
        question = gr.Textbox(
            label="请输入你的问题",
            placeholder="例如：文档里讲了什么内容？这个概念的定义是什么？",
            lines=3,
            scale=8
        )
        submit_btn = gr.Button("提交提问", variant="primary", scale=1)

    answer = gr.Textbox(
        label="AI回答",
        lines=8,
        interactive=False
    )

    submit_btn.click(
        fn=chat,
        inputs=question,
        outputs=answer,
        show_progress="minimal"
    )

    question.submit(
        fn=chat,
        inputs=question,
        outputs=answer,
        show_progress="minimal"
    )

if __name__ == "__main__":
    demo.launch(
        server_port=7680,
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
        debug=True,
        show_error=True
    )
