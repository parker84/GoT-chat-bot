from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from decouple import config
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from streamlit_chat import message
import streamlit as st
import coloredlogs, logging
from tqdm import tqdm
import os

# --------------setup
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', default='INFO'))
st.set_page_config(page_title='GoT Chat', page_icon='⚔️', initial_sidebar_state="auto", menu_items=None)
st.title("Game of Thrones Chat ⚔️")

# st.sidebar.title("Enter Your API Keys 🗝️")
# open_api_key = st.sidebar.text_input(
#     "Open API Key", 
#     value=st.session_state.get('open_api_key', ''),
#     help="Get your API key from https://openai.com/",
#     type='password'
# )
# os.environ["OPENAI_API_KEY"] = open_api_key
# st.session_state['open_api_key'] = open_api_key

@st.cache_data
def create_db_from_txt_files(folder_path):

    all_docs = []
    txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]
    logger.info("Splitting text files into chunks...")
    for filename in tqdm(txt_files):
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path=file_path, autodetect_encoding=True)
        book = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(book)
        all_docs.extend(docs)

    logger.info("Creating database...")
    db = FAISS.from_documents(all_docs, embeddings)
    return db

@st.cache_data
def get_response_from_question(_db, question, memory, k=10):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = _db.similarity_search(question, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Template to use for the system message prompt
    template = """
        Your name is Tyrion Lannister, son of Tywin Lannister, and you are a dwarf.

        You are an expert on the history of Westeros, and seasoned in the art of war and politics.
        
        Here is relevant information on the history of Westeros to 
        help you answer the following questions: {docs}

        Here is relevant information from the current conversation to help you answer the following question: {memory}
        
        Only use the factual information from these books to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "Sorry, I'm a Dwarf not a wizard, I don't know the answer to that".
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = """
        Someone is coming to you for your expert advice, they are in need of your help.
        Here is there question: {question}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=question, docs=docs_page_content, memory=memory)
    return response, docs

if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

question = st.text_input(
    label="Ask Tyrion a question",
    value="Write a battle plan on the best way to attack King's Landing."
)

db = create_db_from_txt_files('./data/got-books')
if len(st.session_state['questions']) > 0:
    memory = '\n\n'.join(
        [
            f'Question: {q}\nAnswer: {a}'
            for q, a in zip(st.session_state['questions'], st.session_state['responses'])
        ]
    )
else:
    memory = None
response, docs = get_response_from_question(db, question=question, memory=memory, k=10)

st.session_state['questions'].append(question)
st.session_state['responses'].append(response)

for i in range(len(st.session_state['questions'])):
    question = st.session_state['questions'][i]
    response = st.session_state['responses'][i]
    message(question, is_user=True)  # align's the message to the left
    message(response, is_user=False)  # align's the message to the right

