import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import urllib.parse

# API Key
api_key = "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"

# Email Contact
support_email = "sourabhsingh4800@gmail.com"  # Replace with your actual support email

def load_documents(directory_path):
    """Load PDF documents from specified directory."""
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def chunk_documents(documents):
    """Chunk documents using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def initialize_rag(directory_path):
    """Initialize the Retrieval-Augmented Generation (RAG) system."""
    # Load documents
    documents = load_documents(directory_path)
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Initialize ChromaDB
    persist_directory = "./chroma_db"
    
    # Create vector store
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    
    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=api_key)
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )
    
    return qa_chain

def streamlit_app():
    """Streamlit application for RAG."""
    st.title("📝 RAG-based Q&A System")
    st.sidebar.header("Settings")
    
    # Directory input
    directory_path = st.sidebar.text_input(
        "Document Directory", value="/home/zealot/RAG_QUICKSHELL/pdf"
    )
    
    # Button to initialize RAG
    if st.sidebar.button("Initialize RAG"):
        st.session_state["qa_chain"] = initialize_rag(directory_path)
        st.success("RAG system initialized successfully!")
    
    # Ensure RAG is initialized
    if "qa_chain" not in st.session_state:
        st.info("Please initialize the RAG system using the sidebar.")
        return
    
    # Query input
    query = st.text_input("Enter your question:")
    if st.button("Submit Query"):
        qa_chain = st.session_state["qa_chain"]
        result = qa_chain({"query": query})
        

        
        # Check if the result contains unclear responses
        if "I'm sorry" in result["result"] or "I didn’t understand" in result["result"]:
            st.subheader("Answer")
            st.write("Sorry, I didn’t understand your question. Do you want to connect with a live agent?")
            email_subject = "Assistance Needed"
            email_body = f"Hi, I need assistance with the following query: {query}"
            mailto_link = f"mailto:{support_email}?subject={urllib.parse.quote(email_subject)}&body={urllib.parse.quote(email_body)}"
            st.markdown(f"[Click here to contact us via Email]({mailto_link})", unsafe_allow_html=True)
        else:
            # Display the result
            st.subheader("Answer")
            st.write(result["result"])

if __name__ == "__main__":
    streamlit_app()
