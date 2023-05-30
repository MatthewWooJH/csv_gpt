# Pre-requisites
import streamlit as st
from streamlit_chat import message
import tiktoken
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS

# LLM-related settings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

st.title("OpenAI Chatbot with CSV reader")
st.text("Insert your API key and a csv file to begin. You can upload up to 3 csv files.")

user_api_key = st.sidebar.text_input(
    label="#### API Key",
    placeholder="Insert API key here",
    type="password")

# Multiple uploads
uploaded_files = [st.sidebar.file_uploader("Upload 1"), st.sidebar.file_uploader("Upload 2"), st.sidebar.file_uploader("Upload 3")]
loaders = []
data = []

for uploaded_file in uploaded_files:
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        loaders.append(loader)

for loader in loaders:
    data.extend(loader.load())

embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
vectors = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=user_api_key),retriever=vectors.as_retriever())

def conversational_chat(query):
    enc = tiktoken.get_encoding("p50k_base")
    query_tokens = enc.encode(query)
    query_token_count = len(query_tokens)

    result = chain({"question": query, "chat_history": st.session_state['history']})
    response_tokens = enc.encode(result["answer"])
    response_token_count = len(response_tokens)

    total_token_count = query_token_count + response_token_count

    st.session_state['history'].append((query, result["answer"]))

    return result["answer"], total_token_count

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything."]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Start chat"]
    
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Insert a prompt", placeholder="Prompt", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output, total_token_count = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        # Display token count in the sidebar
        st.sidebar.text("Total Token Count: {}".format(total_token_count))
        #st.sidebar.text("Response token count: {}".format(total_token_count))
        

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                




#streamlit run tuto_chatbot_csv.py