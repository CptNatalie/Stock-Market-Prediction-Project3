__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
# from apikey import apikey
import openai
import streamlit as st
from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
#from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv


# os.environ["OPENAI_API_KEY"] = ''
# Load environment variables.
# load_dotenv()

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"
# Store the API key in a variable.
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Access your API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

st.title('Chat with Document')
uploaded_file = st.file_uploader('Upload a file:',type=['pdf','docx','txt'])
add_file = st.button('Add File', on_click=clear_history)

vector_store_path='my_vector_store.db'


if uploaded_file and add_file:
    with st.spinner('Reading, chunking and embedding file...'):
        bytes_data = uploaded_file.read()
        file_name = os.path.join('./', uploaded_file.name)
        with open(file_name,'wb') as f:
            f.write(bytes_data)

        name, extension = os.path.splitext(file_name)

        if extension == '.pdf':
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_name)
        elif extension == '.docx':
            from langchain.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_name)
        elif extension == '.txt':
            from langchain.document_loaders import TextLoader 
            loader = TextLoader(file_name)
        else:
            st.write('Document format is not supported!')                 
        
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #\n\n, \n, ' '

        chunks = text_splitter.split_documents(documents)    

        embeddings = OpenAIEmbeddings()

        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='db')
        # Note: Ensure the directory exists and you have write permissions.
        

        #llm = OpenAI(temperature=0)
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

        retriever=vector_store.as_retriever()

        #chain = RetrievalQA.from_chain_type(llm, retriever=retriever )
        crc = ConversationalRetrievalChain.from_llm(llm,retriever)
        st.session_state.crc = crc
        st.success('File uploaded, chunked and embedded successfully')

question = st.text_input('Input your question')

if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        response = crc.run({'question':question,'chat_history':st.session_state['history']})

        st.session_state['history'].append((question,response))
        st.write(response)

        #st.write(st.session_state['history'])
        for prompts in st.session_state['history']:
            st.write("Question: " + prompts[0])
            st.write("Answer: " + prompts[1])    
