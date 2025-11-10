"""
PDF Question-Answering System using LangChain, OpenAI, and Pinecone
This application allows users to query PDF documents using natural language.
"""

import streamlit as st
import os
import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Pinecone
from langchain.chains import load_qa_chain

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION AND INITIALIZATION
# ============================================================================

# Initialize Pinecone connection
def initialize_pinecone():
    """Initialize and return Pinecone connection with API credentials."""
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
    )
    return pinecone

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def load_pdf_documents(directory_path):
    """
    Load all PDF documents from the specified directory.
    
    Args:
        directory_path (str): Path to the directory containing PDF files
        
    Returns:
        list: List of loaded document objects
    """
    pdf_loader = PyPDFDirectoryLoader(directory_path)
    loaded_documents = pdf_loader.load()
    return loaded_documents


def split_documents_into_chunks(documents, chunk_size=800, chunk_overlap=50):
    """
    Split documents into smaller chunks for better processing and retrieval.
    
    Args:
        documents (list): List of document objects to split
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    document_chunks = text_splitter.split_documents(documents)
    return document_chunks


def create_embeddings():
    """
    Create OpenAI embeddings instance for vectorizing text.
    
    Returns:
        OpenAIEmbeddings: Embeddings instance
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    return embeddings


def initialize_vector_store(documents, embeddings, index_name):
    """
    Create or connect to a Pinecone vector store and upload documents.
    
    Args:
        documents (list): List of document chunks to index
        embeddings: Embeddings instance for vectorization
        index_name (str): Name of the Pinecone index
        
    Returns:
        Pinecone: Vector store instance
    """
    vector_store = Pinecone.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )
    return vector_store

# ============================================================================
# QUERY AND RETRIEVAL FUNCTIONS
# ============================================================================

def search_similar_documents(vector_store, user_query, num_results=2):
    """
    Search for documents similar to the user's query using cosine similarity.
    
    Args:
        vector_store: Pinecone vector store instance
        user_query (str): The user's question
        num_results (int): Number of similar documents to retrieve
        
    Returns:
        list: List of similar document chunks
    """
    similar_documents = vector_store.similarity_search(
        query=user_query,
        k=num_results
    )
    return similar_documents


def get_answer_from_documents(user_query, similar_documents, language_model, qa_chain):
    """
    Generate an answer to the user's query based on retrieved documents.
    
    Args:
        user_query (str): The user's question
        similar_documents (list): Relevant document chunks
        language_model: OpenAI LLM instance
        qa_chain: Question-answering chain instance
        
    Returns:
        str: Generated answer
    """
    answer = qa_chain.run(
        input_documents=similar_documents,
        question=user_query
    )
    return answer

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="PDF Q&A Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 3rem;
        }
        .answer-box {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin-top: 1rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #1565c0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📚 PDF Question-Answering Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your PDF documents and get intelligent answers powered by AI</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Document directory input
        documents_directory = st.text_input(
            "Documents Directory",
            value="documents/",
            help="Path to the directory containing PDF files"
        )
        
        # Index name input
        pinecone_index_name = st.text_input(
            "Pinecone Index Name",
            value="langchainvector",
            help="Name of your Pinecone index"
        )
        
        # Model configuration
        st.subheader("Model Settings")
        model_name = st.selectbox(
            "OpenAI Model",
            options=["text-davinci-003", "gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="Select the OpenAI model to use"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Controls randomness in responses (0 = deterministic, 1 = creative)"
        )
        
        num_retrieval_results = st.slider(
            "Number of Documents to Retrieve",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of similar documents to use for answering"
        )
        
        # Initialize button
        initialize_button = st.button("🔄 Initialize System", type="primary")
    
    # Initialize system
    if initialize_button or 'vector_store' not in st.session_state:
        with st.spinner("Initializing system... This may take a few moments."):
            try:
                # Step 1: Load documents
                st.info("📄 Loading PDF documents...")
                loaded_documents = load_pdf_documents(documents_directory)
                
                if not loaded_documents:
                    st.error(f"No documents found in {documents_directory}")
                    st.stop()
                
                st.success(f"✅ Loaded {len(loaded_documents)} document(s)")
                
                # Step 2: Split into chunks
                st.info("✂️ Splitting documents into chunks...")
                document_chunks = split_documents_into_chunks(loaded_documents)
                st.success(f"✅ Created {len(document_chunks)} document chunks")
                
                # Step 3: Initialize Pinecone
                st.info("🌲 Connecting to Pinecone...")
                initialize_pinecone()
                
                # Step 4: Create embeddings
                st.info("🔢 Creating embeddings...")
                embeddings = create_embeddings()
                
                # Step 5: Create vector store
                st.info("💾 Indexing documents in vector store...")
                vector_store = initialize_vector_store(
                    document_chunks,
                    embeddings,
                    pinecone_index_name
                )
                
                # Step 6: Initialize LLM and QA chain
                st.info("🤖 Initializing language model...")
                openai_api_key = os.getenv('OPENAI_API_KEY')
                language_model = OpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    openai_api_key=openai_api_key
                )
                qa_chain = load_qa_chain(language_model, chain_type="stuff")
                
                # Store in session state
                st.session_state['vector_store'] = vector_store
                st.session_state['qa_chain'] = qa_chain
                st.session_state['language_model'] = language_model
                st.session_state['num_retrieval_results'] = num_retrieval_results
                
                st.success("🎉 System initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during initialization: {str(e)}")
                st.stop()
    
    # Main query interface
    st.markdown("---")
    st.header("💬 Ask a Question")
    
    # Query input
    user_query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What are the main topics discussed in the document?",
        help="Type your question about the PDF documents here"
    )
    
    # Query button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and user_query:
        if 'vector_store' not in st.session_state:
            st.warning("⚠️ Please initialize the system first using the sidebar.")
        else:
            with st.spinner("🔍 Searching for relevant information..."):
                try:
                    # Retrieve similar documents
                    similar_documents = search_similar_documents(
                        st.session_state['vector_store'],
                        user_query,
                        st.session_state['num_retrieval_results']
                    )
                    
                    # Generate answer
                    with st.spinner("🤖 Generating answer..."):
                        answer = get_answer_from_documents(
                            user_query,
                            similar_documents,
                            st.session_state['language_model'],
                            st.session_state['qa_chain']
                        )
                    
                    # Display answer
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    
                    # Show source documents
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(similar_documents, 1):
                            st.markdown(f"**Document {i}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
    
    elif submit_button and not user_query:
        st.warning("⚠️ Please enter a question before submitting.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "Powered by LangChain, OpenAI, and Pinecone | "
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
