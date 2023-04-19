import streamlit as st
from langchain.document_loaders import GitbookLoader, WebBaseLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_area("You: ", st.session_state["input"], key="input",
                              placeholder="Type crypto query or 'Dan's office robot greetings' for a robot hello moment",
                              label_visibility='hidden')
    return input_text


# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()


def do_nothing():
    """
    Callback to do nothing when st widgets change
    """
    pass


def add_loader_to_session_state(loader):
    """
    Adds a document loader to the session state.
    """
    st.session_state["loaders"].append(loader)


def document_pipeline():
    """
    Loads documents and pre-processes them for user query
    """
    all_documents = []
    # Add conditionals for different loaders
    for loader in set(st.session_state["loaders"]):
        if isinstance(loader, GitbookLoader) or isinstance(loader, WebBaseLoader):
            documents = loader.load()  # Call the .load() method for GitbookLoader and WebBaseLoader
            all_documents.extend(documents)
        elif isinstance(loader, OnlinePDFLoader):
            documents = loader.load()  # Call the .load_and_split() method for PyPDFLoader
            all_documents.extend(documents)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(all_documents)

    # Generate embeddings for each documents
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_api_key"])

    # Add embeddings to Chroma's vector store
    docsearch = Chroma.from_documents(texts, embeddings)

    return docsearch


def create_vectorstore(loader):
    """
    Loads documents and pre-processes them for user query
    """
    if isinstance(loader, GitbookLoader) or isinstance(loader, WebBaseLoader):
        documents = loader.load()  # Call the .load() method for GitbookLoader and WebBaseLoader
    elif isinstance(loader, OnlinePDFLoader):
        documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=documents)

    # Generate embeddings for each documents
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_api_key"])

    # Add embeddings to Chroma's vector store
    docsearch = Chroma.from_documents(texts, embeddings)

    return docsearch
