import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

# Load environment variables (e.g., Hugging Face API token)
load_dotenv(find_dotenv())

# Define paths for the FAISS database and data folder
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"

# Function to load PDF files from the specified folder
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split documents into smaller chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Function to load the Hugging Face embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Function to create and save a FAISS vector database
def create_vector_database(data_path, db_path):
    documents = load_pdf_files(data_path)  # Step 1: Load PDFs
    text_chunks = create_chunks(documents)  # Step 2: Create text chunks
    embedding_model = get_embedding_model()  # Step 3: Load embedding model
    db = FAISS.from_documents(text_chunks, embedding_model)  # Step 4: Create FAISS DB
    db.save_local(db_path)  # Save database locally
    return len(text_chunks)

# Function to load an existing FAISS vector database
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Function to define a custom prompt for the chatbot
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Function to load the language model (LLM) from Hugging Face
def load_llm(huggingface_repo_id, HUGGINGFACEHUB_API_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HUGGINGFACEHUB_API_TOKEN, "max_length": "512"}
    )
    return llm

# Main function to define the Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title="Ask Medical Chatbot", page_icon=":medical_symbol:")

    # App title
    st.title("Ask Medical Chatbot! :medical_symbol:")

    # Sidebar for database management
    st.sidebar.title("Database Options")
    st.sidebar.markdown(
        """
        **Instructions for Creating a New Vector Database:**
        - Upload your medical encyclopedia PDFs to the `data/` folder in the project directory.
        - Ensure the PDFs contain relevant medical information for the chatbot to use.
        - After uploading, click the **"Create New Vector Database"** button below to process the documents.
        - This will create a vectorized representation of your documents for efficient retrieval during chat.
        - **Note:** Processing may take a few minutes depending on the size of the PDFs.
        """
    )

    # Button to create a new vector database
    if st.sidebar.button("Create New Vector Database"):
        with st.spinner("Creating vector database..."):
            try:
                chunk_count = create_vector_database(data_path=DATA_PATH, db_path=DB_FAISS_PATH)
                st.sidebar.success(f"Vector database created successfully with {chunk_count} chunks!")
            except Exception as e:
                st.sidebar.error(f"Error creating vector database: {e}")

    # Main chatbot UI description
    st.markdown(
        """
        Welcome to the Medical Chatbot! Ask any medical-related question, and I'll provide answers based on the provided context.
        Please ensure your questions are concise and relevant.
        """
    )
    st.divider()

    # Initialize session state for storing chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message("user").markdown(f"**You:** {message['content']}")
        else:
            st.chat_message("assistant").markdown(f"**Chatbot:** {message['content']}")

    # User input for a new query
    prompt = st.chat_input("Enter your question here...")

    if prompt:
        st.chat_message("user").markdown(f"**You:** {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Define the custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
            You are an AI assistant. You have access *only* to the information in the context below. 
            Answer the user's question *strictly* using the context. 

            - If the answer can be found in the context, provide a concise and direct response.
            - If the answer cannot be determined from the context, respond with "I don't know."
            - Do not include any information that is not present in the context.
            - Provide no personal opinions, speculation, or external knowledge.

            Context:
            {context}

            Question:
            {question}

            Start the answer directly. No small talk, please.
            """

        
        # Hugging Face model configuration
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        try:
            vectorstore = get_vectorstore()  # Load vectorstore
            if vectorstore is None:
                st.error("Failed to load the vector store.")

            # Create a RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HUGGINGFACEHUB_API_TOKEN=HUGGINGFACEHUB_API_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 10}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Generate response
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Display chatbot's response
            st.chat_message("assistant").markdown(f"**Chatbot:** {result}")
            # Display source documents in an expandable section
            with st.expander("Source Documents"):
                st.write(source_documents)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    main()
