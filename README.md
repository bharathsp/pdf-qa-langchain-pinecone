# PDF Question-Answering System with LangChain, OpenAI, and Pinecone

A powerful Streamlit-based application that enables natural language question-answering over PDF documents using LangChain, OpenAI embeddings, and Pinecone vector database.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## ✨ Features

- **PDF Document Processing**: Automatically loads and processes PDF documents from a directory
- **Intelligent Chunking**: Splits documents into optimal chunks for better retrieval
- **Vector Search**: Uses OpenAI embeddings and Pinecone for semantic similarity search
- **Natural Language Q&A**: Ask questions in plain English and get accurate answers
- **Source Citation**: View the source documents used to generate each answer
- **Interactive UI**: Beautiful Streamlit interface with real-time feedback
- **Multiple Model Support**: Choose from various OpenAI models (GPT-3.5, GPT-4, etc.)
- **Configurable Parameters**: Adjust temperature, chunk size, and retrieval settings

## 🏗️ Architecture

```
PDF Documents
    ↓
[PyPDFDirectoryLoader] → Load PDFs
    ↓
[RecursiveCharacterTextSplitter] → Split into chunks
    ↓
[OpenAI Embeddings] → Convert chunks to vectors
    ↓
[Pinecone Vector Store] → Store and index vectors
    ↓
[User Query] → Embed query → Similarity Search
    ↓
[RetrievalQA Chain] → Generate answer with context
    ↓
[Streamlit UI] → Display answer
```

## 📦 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/)
- **Pinecone API Key** - Sign up at [Pinecone](https://www.pinecone.io/)
- **Pinecone Index** - Create an index with:
  - Dimension: **1536** (for OpenAI embeddings)
  - Metric: **cosine**
  - Vector Type: **Dense**

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd pdf-qa-langchain-pinecone
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Pinecone Index

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign in or create an account
3. Navigate to **Indexes** → **Create Index**
4. Configure the index:
   - **Index Name**: `langchainvector` (or your preferred name)
   - **Dimensions**: `1536`
   - **Metric**: `cosine`
   - **Vector Type**: `Dense`
   - Select your preferred **Cloud Provider** and **Region**
   - Choose **Capacity Mode**
5. Click **Create Index**

## ⚙️ Configuration

### Step 1: Create `.env` File

Create a `.env` file in the project root directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Step 2: Prepare PDF Documents

1. Create a `documents/` folder in the project root (or use your preferred directory)
2. Place your PDF files in this directory
3. The application will automatically load all PDFs from this directory

**Example structure:**
```
pdf-qa-langchain-pinecone/
├── documents/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── document3.pdf
├── main.py
├── requirements.txt
├── .env
└── README.md
```

## 🎯 Usage

### Running the Application

1. **Activate your virtual environment** (if not already activated)

2. **Start the Streamlit app:**
   ```bash
   streamlit run main.py
   ```

3. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in the terminal

### Using the Application

1. **Configure Settings** (Sidebar):
   - **Documents Directory**: Path to your PDF folder (default: `documents/`)
   - **Pinecone Index Name**: Your Pinecone index name (default: `langchainvector`)
   - **OpenAI Model**: Select the model to use (GPT-3.5-turbo, GPT-4, etc.)
   - **Temperature**: Adjust creativity (0.0 = deterministic, 1.0 = creative)
   - **Number of Documents to Retrieve**: How many document chunks to use for answering

2. **Initialize System**:
   - Click the **"🔄 Initialize System"** button
   - Wait for the system to:
     - Load PDF documents
     - Split into chunks
     - Connect to Pinecone
     - Create embeddings
     - Index documents
     - Initialize the language model

3. **Ask Questions**:
   - Enter your question in the text area
   - Click **"🔍 Get Answer"**
   - View the answer and source documents

### Example Questions

- "What are the main topics discussed in the document?"
- "Summarize the key findings"
- "What is the methodology used?"
- "Explain the conclusion"
- "What are the recommendations?"

## 🔧 How It Works

### Step-by-Step Process

1. **Document Loading**: 
   - `PyPDFDirectoryLoader` reads all PDF files from the specified directory
   - Extracts text content from each page

2. **Text Chunking**:
   - `RecursiveCharacterTextSplitter` splits documents into smaller chunks
   - Default: 800 characters per chunk with 50 character overlap
   - Ensures context preservation across chunks

3. **Embedding Generation**:
   - OpenAI's embedding model converts text chunks into 1536-dimensional vectors
   - Each vector represents the semantic meaning of the text

4. **Vector Storage**:
   - Vectors are uploaded to Pinecone vector database
   - Indexed for fast similarity search

5. **Query Processing**:
   - User's question is converted to an embedding vector
   - Similarity search finds the most relevant document chunks
   - Top-K chunks are retrieved (default: 2)

6. **Answer Generation**:
   - `RetrievalQA` chain combines retrieved context with the question
   - Language model generates a comprehensive answer
   - Answer is displayed with source citations

## 📁 Project Structure

```
pdf-qa-langchain-pinecone/
│
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
├── .env.example           # Example environment file
│
├── documents/             # PDF documents directory
│   └── *.pdf             # Your PDF files
│
└── venv/                 # Virtual environment (created during setup)
```

## 🐛 Troubleshooting

### Common Issues

**1. "No documents found" error**
- Ensure PDF files are in the specified directory
- Check that the directory path is correct
- Verify PDF files are not corrupted

**2. "PINECONE_API_KEY not found"**
- Check your `.env` file exists in the project root
- Verify the API key is correctly set
- Restart the Streamlit app after updating `.env`

**3. "Index not found" error**
- Verify the index name matches your Pinecone index
- Ensure the index exists in your Pinecone account
- Check that the index has dimension 1536

**4. Import errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using the correct Python version (3.8+)
- Try reinstalling packages: `pip install --upgrade -r requirements.txt`

**5. "Rate limit exceeded"**
- OpenAI API has rate limits on free tier
- Wait a few moments and try again
- Consider upgrading your OpenAI plan

**6. Slow performance**
- Reduce the number of documents to retrieve
- Use a smaller chunk size
- Ensure your Pinecone index is in the same region as your usage

## 🔮 Future Enhancements

Potential improvements for the project:

- [ ] Support for multiple file formats (DOCX, TXT, etc.)
- [ ] Batch processing for large document collections
- [ ] Conversation history and follow-up questions
- [ ] Export answers and citations
- [ ] User authentication and multi-user support
- [ ] Advanced filtering and metadata search
- [ ] Integration with other vector databases
- [ ] Support for local LLMs (Llama, Mistral, etc.)
- [ ] Document upload via UI
- [ ] Answer confidence scoring
- [ ] Multi-language support

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Built with ❤️ using Streamlit, LangChain, OpenAI, and Pinecone**
