# Medical Chatbot with PDF Knowledge

A Streamlit-based AI-powered chatbot that answers medical-related questions by leveraging knowledge from uploaded medical encyclopedia PDFs. The chatbot processes and indexes the PDF data, enabling accurate and efficient responses.

## Features
- Upload and process medical PDFs to create a vectorized knowledge database.
- AI chatbot powered by a Hugging Face language model for natural language understanding.
- Context-aware question answering based on the content of uploaded PDFs.
- Easy-to-use web interface with real-time chat functionality.
- Expandable section for viewing source documents used in the chatbot's answers.

## Technologies Used
- **Streamlit**: For the user interface and app deployment.
- **LangChain**: To create chains for document processing and retrieval.
- **Hugging Face**: For the language model and embeddings.
- **FAISS**: For efficient vector search and document retrieval.
- **PyPDFLoader**: For extracting text from PDF files.
- **Python**: As the primary programming language.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/medical-chatbot-with-pdf-knowledge.git
   cd medical-chatbot-with-pdf-knowledge
   ```

2. **Create and Activate a Virtual Environment**:
   - For Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - For macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the project root directory.
   - Add the following lines to the `.env` file:
     ```
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
     ```
   - Replace `your_huggingface_api_token` with your actual Hugging Face API token. You can obtain the token by signing up on [Hugging Face](https://huggingface.co) and navigating to your account settings.

## Usage

1. **Prepare Medical PDF Files**:
   - Place your medical encyclopedia PDFs in the `data/` folder in the project directory.

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Interact with the Chatbot**:
   - Open the provided local URL in your browser.
   - Use the sidebar to create a vector database from the uploaded PDFs.
   - Ask questions in the chat input box, and the chatbot will respond based on the processed PDFs.

## Troubleshooting

- **Issue**: "Failed to load the vector store."
  - **Solution**: Ensure the `data/` folder contains valid PDF files and that the vector database has been created using the sidebar option.

- **Issue**: "Error creating vector database."
  - **Solution**: Check that all PDFs are in a readable format and not corrupted.

- **Issue**: "Module not found" error during execution.
  - **Solution**: Ensure all dependencies are installed by running `pip install -r requirements.txt`.

- **Issue**: "Missing Hugging Face API token."
  - **Solution**: Ensure your `.env` file is created and correctly configured with the `HUGGINGFACEHUB_API_TOKEN`.

