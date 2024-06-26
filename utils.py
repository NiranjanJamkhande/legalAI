from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_transformers import (LongContextReorder,)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI



llm = OpenAI(api_key='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')




embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
reordering = LongContextReorder()


db2 = FAISS.load_local('legislationrules', embeddings,allow_dangerous_deserialization=True)



### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, db2.as_retriever(search_kwargs={'k': 3}), contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are an assistant for legal advice question-answering tasks. \
Use the following pieces of relevant legislation and rules related to cases to answer the question and provide opinions or arguments regarding the case. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


