# Medical Chatbot with PDF Knowledge

A Streamlit-based AI-powered chatbot that answers medical-related questions by leveraging knowledge from uploaded medical encyclopedia PDFs. The chatbot processes and indexes the PDF data, enabling accurate and efficient responses.

## Features
- Upload and process medical PDFs to create a vectorized knowledge database.
- AI chatbot powered by **Mistral-7B-Instruct-v0.3**, a Hugging Face language model.
- Context-aware question answering based strictly on the content of uploaded PDFs.
- Easy-to-use web interface with real-time chat functionality.
- Expandable section for viewing source documents used in the chatbot's answers.

## Technologies Used
- **Streamlit**: For the user interface and app deployment.
- **LangChain**: To create chains for document processing and retrieval.
- **Hugging Face**: For the Mistral-7B-Instruct-v0.3 model and embeddings.
- **FAISS**: For efficient vector search and document retrieval.
- **PyPDFLoader**: For extracting text from PDF files.
- **Python**: As the primary programming language.

### Key AI Model Details
- **Model Name**: `Mistral-7B-Instruct-v0.3`
- **Provider**: Hugging Face
- **Capabilities**:
  - Supports instruction-tuned tasks for enhanced question-answering.
  - Provides concise and context-aware responses.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` for generating dense vector representations of text.

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
   streamlit run rag_medical_chatbot.py.py
   ```

3. **Interact with the Chatbot**:
   - Open the provided local URL in your browser.
   - Use the sidebar to create a vector database from the uploaded PDFs.
   - Ask questions in the chat input box, and the chatbot will respond based on the processed PDFs.

### Example Responses
1. **Context Available**: The bot answers based on the uploaded PDFs.
   - Question: *"How to treat acne?"*
   - Response: The chatbot provides the relevant treatment information because this knowledge was available in the PDF context.
   - ![Context Example](https://github.com/sahilbishnoi26/ai-medical-chatbot/blob/main/data/qna_context.png)

2. **Context Not Available**: The bot doesn't fabricate answers if the information is not in the PDFs.
   - Question: *"Who is the best football player in the world?"*
   - Response: *"I don't know."* The chatbot appropriately responds with "I don't know" as this information is outside the PDF context.
   - ![No Context Example](https://github.com/sahilbishnoi26/ai-medical-chatbot/blob/main/data/qna_no_context.png)

## Troubleshooting

- **Issue**: "Failed to load the vector store."
  - **Solution**: Ensure the `data/` folder contains valid PDF files and that the vector database has been created using the sidebar option.

- **Issue**: "Error creating vector database."
  - **Solution**: Check that all PDFs are in a readable format and not corrupted.

- **Issue**: "Module not found" error during execution.
  - **Solution**: Ensure all dependencies are installed by running `pip install -r requirements.txt`.

- **Issue**: "Missing Hugging Face API token."
  - **Solution**: Ensure your `.env` file is created and correctly configured with the `HUGGINGFACEHUB_API_TOKEN`.
