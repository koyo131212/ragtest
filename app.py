import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Googleのモデルとエンベディングを使うためにインポート
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- UIの基本設定 ---
st.set_page_config(page_title="My RAG App", layout="wide")
st.title("📄 複数ファイル対応！AIチャット (Public Ver.)")

# --- APIキーの設定 ---
# StreamlitのSecretsからAPIキーを読み込む
try:
    genai.configure(api_key=st.secrets["AIzaSyDiEOgaMNj9ERlC_ROrhBY8W-emWqjQV4s"])
except Exception:
    st.error("エラー: Google APIキーが設定されていません。StreamlitのSecretsに設定してください。")
    st.stop()

# --- モデルの初期化（キャッシュして高速化）---
@st.cache_resource
def load_models():
    # Ollamaの代わりにGoogleのモデルを指定
    llm = GoogleGenerativeAI(model="gemini-1.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return llm, embeddings

llm, embeddings = load_models()

# --- RAGチェーンを構築する関数（中身はほぼ同じ） ---
@st.cache_resource
def create_rag_chain(_file_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_text(_file_content)
    
    vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    template = """
    あなたは親切なアシスタントです。以下のコンテキスト情報だけを使って、質問に日本語で答えてください。

    コンテキスト:
    {context}

    質問: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- UIのメイン部分（変更なし） ---
uploaded_files = st.file_uploader(
    "質問したい内容が書かれたテキストファイル（.txt）を複数選択できます。", 
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.getvalue().decode("utf-8")
        all_text += file_content + "\n\n---\n\n"
    
    rag_chain = create_rag_chain(all_text)

    file_names = [f.name for f in uploaded_files]
    st.success(f"以下のファイルを読み込みました: {', '.join(file_names)}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("質問を入力..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AIが考えています..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("ファイルをアップロードすると、チャットが開始されます。")
