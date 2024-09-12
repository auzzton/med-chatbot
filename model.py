from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl


DBfaisspath="vectorstores/db_faiss"


custom_prompt_template="""Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_pmpt():
    """
    prompt template for qna retrieval for each question

    """
    prompt=PromptTemplate(template=custom_prompt_template,input_varibles=['Context','Question'])

    return prompt


def load_llm():

    llm=CTransformers(model="D:\Medbot llama\llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temprature=0.5)

    return llm

def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(llm=llm,
                                          chain_type='stuff',
                                          retriever=db.as_retriever(search_kwargs={"k":2}),
                                          return_source_documents=True,
                                          chain_type_kwargs={'prompt':prompt}
                                          )
    return qa_chain


def qa_bot():

    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':'cpu'})
    db=FAISS.load_local(DBfaisspath,embeddings,allow_dangerous_deserialization=True)
    llm=load_llm()
    qa_prompt=set_custom_pmpt()
    qa= retrieval_qa_chain(llm,qa_prompt,db)

    return qa

def final_result(query):
    qa_result=qa_bot()
    response=qa_result.invoke({'query':query})
    return response


if __name__=="__main__":
    query=input("enter your question")
    result=final_result(query)
    print(result)
