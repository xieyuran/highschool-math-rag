from rag import qa_chain
import gradio as gr

def chat(question):
    if not question or question.strip() == "":
        return "请输入有效的高中数学问题～"

    try:
        result = qa_chain.invoke({"question": question.strip()})
        return result["result"]
    except Exception as e:
        return f"回答生成失败：{str(e)}\n请检查：1.Ollama是否启动 2.math.txt是否存在 3.qwen:0.5b模型是否下载"

with gr.Blocks(title="高中数学RAG智能答疑系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📘 高中数学RAG智能答疑系统
    基于本地大模型（qwen:0.5b）+ 教材知识点检索，回答严谨、无幻觉
    > 仅根据math.txt中的教材内容回答，不编造、不扩展
    """)

    with gr.Row():
        question = gr.Textbox(
            label="请输入你的高中数学问题",
            placeholder="例如：什么是集合？充分条件和必要条件的区别？",
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
        server_port=7860,  
        server_name="127.0.0.1", 
        inbrowser=True, 
        share=False,  
        debug=True,  
        show_error=True 
    )