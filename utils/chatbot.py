from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HugginFaceHub
from langchain.embeddings import HuggingFaceStructEmbeddings

def create_vectorstore(chunks):
    embeddings = HuggingFaceStructEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore

def create_conversation_chain(vectorstore):
    llm = HugginFaceHub(repo_id='google/flan-t5-large', model_kwargs={"max_length":512, "temperature":0.1})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain