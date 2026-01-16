import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 페이지 설정
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖")
st.title("📄 PDF 기반 AI 챗봇 (Gemini 2.5 Flash)")

# 2. API Key 설정 (Streamlit Secrets 활용)
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    st.error("Secrets 키를 찾을 수 없습니다. .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()

# 3. 리소스 캐싱 (PDF 로딩 및 벡터 DB 생성은 한 번만 실행)
@st.cache_resource
def get_vectorstore():
    pdf_path = "test.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"'{pdf_path}' 파일이 없습니다. 프로젝트 루트에 파일을 위치시켜 주세요.")
        return None

    # PDF 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # 임베딩 생성 (Gemini Embedding)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 벡터 저장소 생성 (FAISS)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# 4. RAG 체인 생성
def get_rag_chain(vectorstore):
    # LLM 설정 (요청하신 gemini-2.5-flash 모델 사용)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # 검색기(Retriever) 설정
    retriever = vectorstore.as_retriever()

    # 프롬프트 템플릿
    system_prompt = (
        "당신은 문서를 기반으로 질문에 답변하는 유능한 AI 어시스턴트입니다. "
        "아래의 제공된 문맥(Context)을 사용하여 질문에 답변하세요. "
        "답을 모르면 모른다고 말하고, 없는 내용을 지어내지 마세요. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 체인 결합
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# --- 메인 로직 실행 ---

# 벡터 DB 초기화
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()

# 벡터 DB 로드 성공 시에만 채팅 인터페이스 표시
if st.session_state.vectorstore is not None:
    
    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("PDF 내용에 대해 질문해주세요..."):
        # 사용자 메시지 표시 및 저장
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("문서를 분석 중입니다..."):
                try:
                    rag_chain = get_rag_chain(st.session_state.vectorstore)
                    response = rag_chain.invoke({"input": prompt})
                    answer = response['answer']
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
