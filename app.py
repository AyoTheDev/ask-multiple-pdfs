import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

from prompts import prompt_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_prompt=prompt_template

    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def test():
    st.write("where does this appear?")
def main():
    load_dotenv()

    gpt4 = ChatOpenAI(model_name='gpt-3.5-turbo-0125')
    prompt = PromptTemplate(template=prompt_template, input_variables=["resume", "job_spec"])
    llm_chain = LLMChain(
        prompt=prompt,
        llm=gpt4
    )
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Executive Summary Generator")
    response=""
    with st.sidebar:
        st.subheader("Your documents")
        cv_doc = st.file_uploader(
            "Upload candidate profile", accept_multiple_files=True)

        job_spec_doc = st.file_uploader(
            "Upload job specification", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                cv_raw_text = get_pdf_text(cv_doc)
                job_spec_text = get_pdf_text(job_spec_doc)
                response = llm_chain.run({'resume': cv_raw_text, 'job_spec': job_spec_text})

    st.write(response)

if __name__ == '__main__':
    main()
