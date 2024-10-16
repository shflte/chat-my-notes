import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import dotenv

# 1. Load Markdown Files
def load_markdown_files(directory):
    # Delete the .obsidian folder if it exists
    obsidian_folder = os.path.join(directory, '.obsidian')
    if os.path.exists(obsidian_folder) and os.path.isdir(obsidian_folder):
        for root, dirs, files in os.walk(obsidian_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(obsidian_folder)

    markdown_texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            with open(os.path.join(directory, filename), 'r') as file:
                markdown_texts.append(file.read())
    return markdown_texts

# 2. Chunk the Text
def chunk_text(texts, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents(texts)

# 3. Create Embeddings and Store in FAISS (re-embed every time)
def embed_all(texts, vector_store_path='vectorstore.faiss'):
    embeddings = OpenAIEmbeddings()
    documents = chunk_text(texts)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(vector_store_path)

# Load and re-embed the markdown files
dotenv.load_dotenv()
markdown_texts = load_markdown_files('vault')
embed_all(markdown_texts)
