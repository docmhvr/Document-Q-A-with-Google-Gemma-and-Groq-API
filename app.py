import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

# Load Groq and Google API key from environment file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Document Q&A with Gemma and Groq")

# LLM model Google's Gemma using Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it" )
print("Hello, I am Gemma!!")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context and reply accurately based on the question
    <context>
    {context}
    <context>
    Questions: {input}

    """
)

# Create vector embeddings from pdf's
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./pdfs") # Data ingestion
        st.session_state.docs = st.session_state.loader.load() # Load Documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

prompt1 = st.text_input("What do you want to ask your documents?")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector DB is Ready!")

if prompt1:
    # A retrieval chain to retrieve info from vector store and llm to chat out the reponse
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Response output to UI
    response = retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    # with streamlit expander
    with st.expander("Document Similarity Search"):
        # Find relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------")


