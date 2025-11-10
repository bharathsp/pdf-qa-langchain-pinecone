# pdf-qa-langchain-pinecone
Using langchain libraries for this project.
Using RecursiveCharacterTextSplitter to convert PDF document into text chunks, Use openai embeddings to convert the text chunks into vectors. And storing it on vector DB Pinecone for querying on top of it to get info from the PDF through similarity search.
Creating frontend using streamlit

Architecture:

Steps:
Create virtual environment
Follow these steps to set up and run the project on your local machine.
Create a Virtual Environment
python3 -m venv venv
Activate the environment:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
venv\Scripts\Activate.ps1

Create requirements.txt
unstructured
tiktoken
pinecone-client
pypdf
openai
langchain
pandas 
numpy
python-dotenv

Install requirements
pip install -r requirements.txt

Create .env file
Add OpenAi API Key

Create main.py file for the actual code
Read PDF using PyPDFDirectoryLoader
Split document into chunks using RecursiveCharacterTextSplitter. chunk_size=800, chunk_overlap=50
Create pinecone DB 
Go to - https://app.pinecone.io/
Login into pinecone > Go to Database > Indexes > Create Index > Index name: langchainvector > Select manual configiration > Vector Type: Dense; Dimension: 1536; Metric: cosine > Select Capacity mode, cloud provider, Region > Create Index > 

Run the Streamlit App
streamlit run main.py
🌐 Open your browser and visit: 👉 http://localhost:8501

You’ll see the Text-to-SQL Query App interface.

🧰 Deactivate Virtual Environment
When done, deactivate your environment:

deactivate

💬 Example Usage
Try these questions in the Streamlit input box:

🧠 Sample Output Snapshots

🧩 How It Works (Step-by-Step)

🧩 Future Enhancements
