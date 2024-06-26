from flask import Flask, render_template, request, redirect, url_for,jsonify,session
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_community.document_transformers import (LongContextReorder,)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from utils import *
import os
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables.history import RunnableWithMessageHistory


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
reordering = LongContextReorder()
db = FAISS.load_local('traindb', embeddings,allow_dangerous_deserialization=True)



def retrieve_similar_contents(user_query,db):
    similar_docs = db.similarity_search(user_query,k=3)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(similar_docs)
    similar_c =  [doc.page_content for doc in reordered_docs]
    return reordered_docs,similar_c


def generate_response(similar_docs, user_query):
    reordered_docs = reordering.transform_documents(similar_docs)

    document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
    document_variable_name = "context"

    llm = OpenAI(api_key='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')
    stuff_prompt_override = """{query}
    -----
    {context}
    """
    prompt = PromptTemplate(
        template=stuff_prompt_override, input_variables=["context", "query"]
    )

    # Instantiate the chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    response = chain.run(input_documents=reordered_docs, query=user_query + """""")
    return response



application = Flask(__name__)
application.config['SECRET_KEY'] = 'HTS'
Bootstrap(application)

# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class QueryForm(FlaskForm):
    user_query = StringField('Enter your query:', validators=[DataRequired()])
    submit = SubmitField('Get Response')

@application.route('/', methods=['GET', 'POST'])
def index():
    form = QueryForm()
    return render_template('index.html', form=form)

def document_to_dict(doc):
    return {
        'content': doc.page_content,
        'metadata': doc.metadata
    }





conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)



# Notice we don't pass in messages. This creates
# a RunnableLambda that takes messages as input

@application.route('/submit_case', methods=['POST'])
def submit_case():
    case_details = request.form.get('case_details', '').strip()
    user_pdf = request.files.get('user_pdf')

    if not case_details and (not user_pdf or user_pdf.filename == ''):
        return jsonify({'message': 'Please enter case details'})

    similar_docs_serializable = []
    pdf_url = None

    if case_details:
        _, similar_contents = retrieve_similar_contents(user_query=case_details, db=db)
        similar_docs_serializable = [document_to_dict(doc) for doc in _]
        session['case_details'] = case_details

    if user_pdf and user_pdf.filename != '':
        if user_pdf.filename.endswith('.pdf'):
            filename = secure_filename(user_pdf.filename)
            filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            user_pdf.save(filepath)

            loader = PyPDFLoader(filepath)
            pages = loader.load()
            case_details = '\n'.join([i.page_content for i in pages])
            _, similar_contents = retrieve_similar_contents(user_query=case_details, db=db)
            similar_docs_serializable = [document_to_dict(doc) for doc in _]
            session['case_details'] = case_details
            print(case_details)

            pdf_url = url_for('submit_case', filename=filename)

    return jsonify({
        'message': 'Case details and PDF uploaded successfully',
        'similar_contents': similar_docs_serializable,
        'pdf_url': case_details
    })



@application.route('/get_decision', methods=['POST'])
def get_decision():
    data = request.json
    user_query = data.get('case_details')

    similar_docs,similar_c = retrieve_similar_contents(user_query,db=db)  # Dummy function, replace with actual logic
    #response = generate_response(similar_docs, user_query)  # Dummy function, replace with actual logic
    response = 'decision'

    r = conversational_rag_chain.invoke({"input": user_query}, config={"configurable": {"session_id": "abc123"}},)["answer"]
    #r = conversational_rag_chain.
    #r = 'answer'
    
    
    similar_docs_serializable = [document_to_dict(doc) for doc in similar_docs]
    return jsonify({
        'decision': response,
        'similar_contents': similar_docs_serializable
    })


from langchain_core.messages import AIMessage

@application.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data.get('user_question')
    current_case_details = session.get('case_details', 'No case details found')
    last_case_details = session.get('last_case_details', None)

    # Determine whether to use case_details + user_question or just user_question
    if current_case_details == last_case_details:
        input_text = user_question
    else:
        input_text = current_case_details + user_question
        # Update last_case_details to current_case_details
        session['last_case_details'] = current_case_details

    with get_openai_callback() as cb:
        r = conversational_rag_chain.invoke({"input": input_text}, config={"configurable": {"session_id": "abc123"}})
        sim = r['context']
        print(sim[0].metadata.keys())
        q_sim = [str(ii.page_content) + '\n' + ' source: ' + str(ii.metadata['source']) for ii in sim]
        for pk in q_sim:
            print(pk)
            print()
    print(cb)
    for message in store["abc123"].messages:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"


    return jsonify({
        'response': r['answer'],
        'similar_contents': q_sim
    })
if __name__ == '__main__':
    application.run(debug=False, host = '0.0.0.0')
