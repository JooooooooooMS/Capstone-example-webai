# from dotenv import load_dotenv
# load_dotenv()
pip install -U langchain-text-splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import tempfile
import os

st.title("PDFReader")
st.write("---")

uploaded_file = st.file_uploader(label="파일 선택", type="pdf")
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(pages, embeddings)

    st.header("PDF를 넣어 문제를 만들어 보세요!")

    if st.button('문제 생성 하기'):
        with st.spinner('Wait for it...'):
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are world class professor of all fields called a walking dictionary."),
                ("user", "{input}")
            ])

            output_parser = StrOutputParser()
            llm = ChatOpenAI()
            chain = prompt | llm | output_parser

            # chain.invoke({"input": "레미제라블의 줄거리를 공백없이 1000자로 써줘"})

            from langchain_community.vectorstores import FAISS
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(pages)
            vector = FAISS.from_documents(documents, embeddings)

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

            response = retrieval_chain.invoke({"input": "에 대한 객관식 문제를 하나, 주관식 문제를 하나 만들어줘"})
            st.write(response["answer"])
