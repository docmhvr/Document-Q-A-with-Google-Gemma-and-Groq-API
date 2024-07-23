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
google_api_key = os.getenv("GOOGLE_API_KEY")

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
    return 0

