import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import uuid

# Initialize the vector store
client = Client(Settings(persist_directory="./vectorstore"))
collection_name = "documents"

# Create or load the collection
existing_collections = client.list_collections()
if collection_name in [coll.name for coll in existing_collections]:
    collection = client.get_collection(collection_name)
else:
    collection = client.create_collection(collection_name)

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the LLM model
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # Lightweight model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm = load_llm()

# Helper Functions
def parse_pdf(file_path):
    """Extract text from PDF."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_embeddings(text):
    """Generate embeddings for a given text."""
    return embedding_model.encode(text, show_progress_bar=True)

def store_embeddings(collection, doc_id, text, embeddings):
    """Store document embeddings in the vector store."""
    collection.add(documents=[text], ids=[doc_id], embeddings=[embeddings])

def truncate_text(text, max_length=1024):
    """Truncate text to a specified maximum length."""
    return text[:max_length]

def query_llm(prompt):
    """Query the LLM model."""
    return llm(prompt, max_new_tokens=50)[0]['generated_text']

def retrieve_and_query(collection, query, llm):
    """Retrieve relevant documents and query the LLM."""
    results = collection.query(query_texts=[query], n_results=3)

    # Flatten and combine the retrieved documents
    relevant_docs = " ".join(
        " ".join(doc) if isinstance(doc, list) else doc
        for doc in results['documents']
    )

    # Truncate to prevent memory overload
    relevant_docs = truncate_text(relevant_docs)

    # Query the LLM
    prompt = f"Based on the following documents: {relevant_docs}, answer: {query}"
    return query_llm(prompt)

# Streamlit Interface
st.title("Content Engine")
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.write("Processing documents...")
    
    # Process uploaded PDFs
    docs = {file.name: parse_pdf(file) for file in uploaded_files}
    
    # Generate and store embeddings
    st.write("Generating embeddings...")
    for name, text in docs.items():
        embeddings = generate_embeddings(text)
        store_embeddings(collection, name, text, embeddings)
    
    st.write("System is ready for queries.")
    
    # Query interface
    query = st.text_input("Enter your query:")
    if query:
        st.write("Retrieving insights...")
        try:
            response = retrieve_and_query(collection, query, llm)
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
