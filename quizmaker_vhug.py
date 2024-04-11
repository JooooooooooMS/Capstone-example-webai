# from dotenv import load_dotenv
# load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st
import tempfile
import os

st.title("AI Quiz Maker with Capstone")
st.write("---")
st.header("입력 유형을 선택하세요!")
option1 = st.selectbox('하나를 고르세요.', ['pdf', 'text', 'image'])
if (option1 == 'pdf'):
    uploaded_file = st.file_uploader(label="파일 선택", type="pdf")
elif (option1 == 'text'):
    uploaded_file = st.text_area(label="Text to analyze", value=None)
elif (option1 == 'image'):
    uploaded_file = st.file_uploader(label='이미지 선택', type=['png', 'jpg', 'jpeg'])
else:st.write("ERROR: You Don't Select The Type Of Problem")

st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

def image_to_document(uploaded_file):
    loader = UnstructuredImageLoader(uploaded_file, mode="elements")
    data = loader.load()
    return data

def text_to_document(uploaded_file):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([uploaded_file])
    return texts

if uploaded_file is not None:
    if (option1 == 'pdf'):
        pages = pdf_to_document(uploaded_file)
    elif (option1 == 'text'):
        pages = text_to_document(uploaded_file)
    elif (option1 == 'image'):
        pages = image_to_document(uploaded_file)
    else:st.write("ERROR: You Don't Select The Type Of Problem")

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(pages, embeddings)

    st.header("문제 유형을 선택하세요!")
    option = st.selectbox('하나를 고르세요.', ['객관식', 'OX퀴즈', '주관식'])

    if st.button('문제 생성 하기'):
        with st.spinner('Wait for it...'):
            # prompt = ChatPromptTemplate.from_messages([
            #     ("system", "You are world class professor of all fields called a walking dictionary."),
            #     ("user", "{input}")
            # ])

            # output_parser = StrOutputParser()
            llm = ChatOpenAI()
            # chain = prompt | llm | output_parser

            # chain.invoke({"input": "레미제라블의 줄거리를 공백없이 1000자로 써줘"})


            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(pages)
            vector = FAISS.from_documents(documents, embeddings)
            # elif (option1 == 'image'):
            #     documents = text_splitter.split_documents(pages)
            #     vector = FAISS.from_documents(documents, embeddings)
            # else:
            #     documents =  text_splitter.split_text(pages)
            #     vector = FAISS.from_image(documents, embeddings)

            from langchain.chains.combine_documents import create_stuff_documents_chain

            prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

                <context>
                {context}
                </context>

                Question: {input}""")

            document_chain = create_stuff_documents_chain(llm, prompt)

            from langchain_core.documents import Document

            # document_chain.invoke({
            #     "input": "레미제라블의 줄거리는 뭐야?",
            #     "context": [Document(page_content="langsmith can let you visualize test results")]
            # })

            from langchain.chains import create_retrieval_chain

            retriever = vector.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            if (option == '객관식') :
                response = retrieval_chain.invoke(
                    {"input": "에 대해 문제를 \"problem:다음 보기중 옳은 것은?\n 1번.***\n 2번.***\n 3번.***\n 4번.***\n 5번.**\"과 같이 질문과 그에 대한 정답1개, 오답4개의 보기를 포함하는 형식으로 만들어주고 그에 대한 정답 번호를 \"answer:\"에다가 써줘 "})
            elif (option == '주관식') :
                response = retrieval_chain.invoke({"input": "에 대한 서술형 문제를 만들어서 \"problem:\" 다음에 써주고 그에 대한 답을 \"answer:\"에다가 써줘 "})
            elif (option == 'OX퀴즈'):
                response = retrieval_chain.invoke({"input": "에 대한  문제를 \"problem:다음중 참인 문장은 A.*** B.***\"과 같은 질문과 그에 대한 정답인 문장 혹은 보기와 오답인 문장 혹은 보기를 포함하는 형식으로 만들어주고 그에 대해 어느쪽이 답인지 \"answer:\"에다가 써줘 "})
            else:
                resource = "ERROR: You Don't Select The Type Of Problem"
            # response = retrieval_chain.invoke({"input": "에 대한 " + option +"문제를 만들어서 \"문제:\" 다음에 써주고 그에 대한 답을 \"답:\"에다가 써줘 "})
            st.write(response["answer"])
