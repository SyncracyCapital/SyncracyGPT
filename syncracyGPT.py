import glob
import random

import streamlit as st
from PIL import Image
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import GitbookLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader

from utils import get_text, document_pipeline, add_loader_to_session_state, do_nothing
from utils import new_chat

# Set Streamlit page configuration
st.set_page_config(
    page_title="SyncracyGPT",
    page_icon="🤖",
    layout="wide",
)

markdown = """<h1 style='font-family: Calibri; text-align: center;'><img 
src="https://images.squarespace-cdn.com/content/v1/63857484f91d71181b02f971/9943adcc-5e69-489f-b4a8-158f20fe0619
/Snycracy_WebLogo.png?format=500w" alt="logo"/>GPT</h1>"""

st.markdown(markdown, unsafe_allow_html=True)

# ------------------------ Set Initial Session State ------------------------ #

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "sources" not in st.session_state:
    st.session_state["sources"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "loaders" not in st.session_state:
    st.session_state["loaders"] = []
if "training_model_status" not in st.session_state:
    st.session_state["training_model_status"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "document_db" not in st.session_state:
    st.session_state["document_db"] = False
if "load_button_clicked" not in st.session_state:
    st.session_state["load_button_clicked"] = False
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = False

# ------------------------ Set up sidebar with various options ------------------------ #

# Set up sidebar with various options
with st.sidebar.expander("📈💻 Research Documents", expanded=False):
    loader_map = {'Gitbook': GitbookLoader,
                  'URL': WebBaseLoader}

    document_type = st.selectbox(label='Document Type', options=loader_map.keys(),
                                 on_change=do_nothing)

    if document_type == 'Gitbook':
        gitbook_url = st.text_input('Enter Gitbook URL', on_change=do_nothing)

    elif document_type == 'URL':
        urls_input = st.text_input("Enter URLs", on_change=do_nothing)

    elif document_type == 'PDF':
        pdf_file = st.text_input('Enter PDF URL', on_change=do_nothing)

    document_description = st.text_area('Enter Document Description', on_change=do_nothing)

    load_docs, train_model = st.columns(2)

    with load_docs:
        loaded_doc = st.button('Load Documents')
        if loaded_doc:
            if document_type == 'Gitbook' and gitbook_url:
                st.session_state["load_button_clicked"] = True
            elif document_type == 'URL' and urls_input:
                st.session_state["load_button_clicked"] = True
            elif document_type == 'PDF' and pdf_file:
                st.session_state["load_button_clicked"] = True
            else:
                st.warning('Please provide the necessary input (URL or file) for the selected document type')

        if st.session_state["load_button_clicked"]:
            try:
                document_loader = loader_map[document_type]
                if document_type == 'Gitbook' and gitbook_url:
                    document_loader = GitbookLoader(gitbook_url, load_all_paths=True)
                elif document_type == 'URL' and urls_input:
                    document_loader = WebBaseLoader(urls_input)
                elif document_type == 'PDF' and pdf_file:
                    document_loader = PyPDFLoader(pdf_file)

                add_loader_to_session_state(document_loader)
                st.session_state["load_button_clicked"] = False  # Reset the flag
            except NameError:
                st.error('Please select a document type and enter the corresponding URL/paths')

    if len(st.session_state["loaders"]) < 1:
        st.warning('Please load at least one document')
    else:
        st.success(f'Successfully loaded {len(st.session_state["loaders"])} documents')

    with train_model:
        if st.button('Train SyncracyGPT'):
            with st.spinner('Training SyncracyGPT....'):
                document_db = document_pipeline()
                st.session_state['document_db'] = document_db
                st.session_state['training_model_status'] = True

with st.sidebar.expander("🛠️ Model Settings", expanded=False):
    MODEL = st.selectbox(label='Model',
                         options=['gpt-3.5-turbo', 'text-davinci-003',
                                  'text-davinci-002', 'code-davinci-002'])
    TEMPERATURE = st.slider(label='Temperature', min_value=0.0, max_value=1.0, value=0.0)

API_O = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]",
                              placeholder="Paste your OpenAI API key here (sk-...)",
                              type="password")
st.session_state["openai_api_key"] = API_O
if st.session_state["openai_api_key"]:
    st.sidebar.success('OpenAI API key successfully set')

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# ------------------------ User interface ------------------------ #

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if st.session_state['training_model_status']:
    st.success('SyncracyGPT is ready to chat!')

if user_input:

    # Check if the user has entered their OpenAI API key
    if not st.session_state["openai_api_key"]:
        st.error('Please enter your OpenAI API key')
        st.stop()

    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=TEMPERATURE,
                     openai_api_key=st.session_state["openai_api_key"],
                     streaming=True,
                     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                     verbose=True,
                     model_name=MODEL)

    # Create the ConversationChain object with the specified configuration
    try:
        chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                      retriever=st.session_state.document_db.as_retriever(
                                                          search_kwargs={"k": 1}),
                                                      return_source_documents=True)
    except AttributeError:
        st.error('Please train SyncracyGPT before chatting')
        st.stop()

    if user_input == "Dan's office robot greetings":
        images = []
        for img in glob.glob("dangreetings/*.jpeg"):
            images.append(Image.open(img))
        st.image(random.choice(images), caption="Dan's office robot greetings")
        st.stop()

    output = chain({"question": user_input, "chat_history": st.session_state.chat_history})

    source_links = '\n'.join([doc.metadata['source'] for doc in output['source_documents']])

    output_text = f"{output['answer']} \n\nSources: \n\n{source_links}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output_text)
    st.session_state.chat_history.append((user_input, output['answer']))

# Display the conversation history using an expander, and allow the user to download it
download_str = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
