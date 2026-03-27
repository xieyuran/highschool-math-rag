import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


with open("math.txt", "r", encoding="utf-8") as f:
    content = f.read()

documents = [Document(page_content=content)]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "；", "！", "？"],
    strip_whitespace=True,
)
chunks = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


VECTOR_DB_PATH = "./vector_db"

if os.path.exists(VECTOR_DB_PATH) and len(os.listdir(VECTOR_DB_PATH)) > 1:
    vectordb = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding
    )
else:
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=VECTOR_DB_PATH
    )

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

llm = OllamaLLM(
    model="qwen:0.5b",
    temperature=0.0,
    max_tokens=256,
    top_p=0.1
)


prompt = ChatPromptTemplate.from_template("""
你是专业高中数学答疑AI，必须严格遵守规则：
1. 只依据math.txt内容回答
2. 不知道就说：教材中未提及
3. 不编造、不扩展、不举例、不解释多余内容
4. 回答简洁、严谨、一字不多

教材内容：
{context}

问题：
{question}
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False,
    input_key="question"
)

if __name__ == "__main__":
    pass