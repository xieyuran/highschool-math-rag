# 通用本地知识库问答系统
基于 RAG 检索增强生成的本地知识库问答系统，支持任意 TXT 文档离线问答，严格依据文档内容回答，无幻觉、不编造，可直接用于个人知识库、企业文档、学习资料等场景。

## 项目简介
本项目基于 Python + LangChain + Chroma + Ollama 实现轻量级本地 RAG 系统，无需联网即可完成文档加载、文本分块、向量化、语义检索与智能问答，提供简洁的 Web 交互界面，开箱即用。

## 技术栈
- Python
- LangChain
- Chroma 向量数据库
- BAAI/bge-small-zh-v1.5 中文向量模型
- Ollama + qwen:0.5b 本地大模型
- Gradio 交互界面

## 文件结构
- math.txt             示例知识库（高中数学知识点）
- rag.py               RAG 核心逻辑
- main.py              Gradio 网页交互入口
- requirements.txt     Python 依赖
- README.md            项目说明

## 运行步骤
1. 安装依赖
pip install -r requirements.txt

2. 下载模型
ollama pull qwen:0.5b

3. 启动系统
python main.py

4. 访问界面
http://127.0.0.1:7860

## 功能特点
- 支持任意 TXT 文档作为知识库
- 全程本地离线运行
- 严格依据文档回答，不编造、不扩展
- 响应快速，轻量化部署
- 界面简洁易用，开箱即用

## 使用说明
将需要问答的文档上传即可实现对应领域的智能问答，支持法律、金融、学术、企业文档、小说、手册等任意纯文本场景。
