# Standard library imports
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

# Langchain modules for building the retrieval-augmented generation (RAG) pipeline
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain, StuffDocumentsChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Language model and embeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store for persistent retrieval
from langchain_chroma import Chroma

from prompt import PROMPT, EXAMPLE_PROMPT

# Load environment variables (e.g., API keys in the .env file)
load_dotenv()

# Configuration Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore" # Creates a resources directory in the same file path where this .py file is stored
COLLECTION_NAME = "real_estate"

# Global variables
llm = None
vector_store = None

def initialize_components():
    """
    Initializes the LLM (ChatGroq) and Vector Store (Chroma) with embeddings.
    Ensures we only load these heavy resources once.
    """
    global llm, vector_store

    if llm is None: # Prevents duplication and re-run if already exists
        # Create the Groq LLM interface
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500
        )

    if vector_store is None: # Prevents duplication if already exists
        # Create the embedding model
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code":True}
        )
        # Initialize Chroma vector store (persistent local storage)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

# PROCESS URLS AND STORE IN VECTOR DB
def process_urls(urls):

    """
    - Scrapes text content from given URLs.
    - Splits the text into manageable chunks.
    - Stores embeddings in the local Chroma vector database.

    :param urls: List of website URLs to ingest
    """
    yield "Initializing components"
    initialize_components()

    # WARNING: This wipes and resets the collection
    yield "Resetting vector store...✅"
    vector_store.reset_collection() # Empty the database

    yield "Loading data...✅"
    # Use custom User-Agent header to avoid simple anti-bot blocking and protections
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )
    data = loader.load() # Loads raw HTML and extracts text

    yield "Splitting text into chunks...✅"
    # Split the raw text into smaller chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size = CHUNK_SIZE,
    )
    docs = text_splitter.split_documents(data) # docs are the chunks which we insert into the vector db

    yield "Add chunks to vector database...✅"
    # Generate unique IDs for each chunk
    uuids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(docs, ids=uuids) # Store documents in the Chroma vector store

    yield "Done adding docs to vector database...✅"

# GENERATE ANSWER FROM VECTOR DB
def generate_answer(query):
    """
    - Given a question, uses Retrieval-Augmented Generation (RAG) to answer.
    - Retrieves relevant text chunks from the vector store.
    - Calls the LLM to synthesize an answer from retrieved context.

    :param query: User's question in plain text
    :return: Tuple of (answer text, sources used)
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    # Create an LLMChain using our language model and the custom QA prompt template
    # This chain takes a question + context and generates an answer
    llm_chain = LLMChain(
        llm=llm,
        prompt=PROMPT
    )

    # Wrap the LLMChain in a StuffDocumentsChain
    # This chain is responsible for taking the retrieved document chunks,
    # formatting them using EXAMPLE_PROMPT, and feeding the combined context to the LLM
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, # The LLMChain with our main PROMPT
        document_prompt=EXAMPLE_PROMPT, # How to format individual documents
        document_variable_name="summaries" # The variable name used in the PROMPT
    )

    # Create the RetrievalQAWithSourcesChain
    # This ties together the retriever (which fetches relevant chunks from vector DB)
    # and the StuffDocumentsChain (which generates the final answer from those chunks)
    chain = RetrievalQAWithSourcesChain(
        retriever=vector_store.as_retriever(), # The retriever from vector store
        combine_documents_chain=stuff_chain, # Our custom StuffDocumentsChain
        return_source_documents=True # Include the sources in the final answer
    )

    # Invoke the chain to get the answer
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "") # Returns the source article from which the answer was retrieved

    return result['answer'], sources

if __name__ == "__main__":
    urls = [
        "https://www.nbcnews.com/business/economy/why-federal-reserve-is-keeping-interest-elevated-right-now-rcna213821",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    # Scrape, split, embed, and store the content
    process_urls(urls)

    # Answer a question using the newly indexed knowledge
    answer, sources = generate_answer("Can you tell me what was the 30 year fixed mortgage rate along with the date?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")