import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Path to your FAISS index
DB_FAISS_PATH = 'vectorstores/db_faiss'

# Custom prompt template for question answering
custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Provide a detailed and helpful answer below:
Helpful answer:
"""

# Setup custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=False,  # We don't need source documents to speed up response
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Load the language model (CTransformers with Llama)
def load_llm():
    llm = CTransformers(
        model="D:/python medical chatbot/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=1024,  # Increase this if you need longer responses
        temperature=0.7
    )
    return llm

# QA Model Function
def qa_bot():
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    # Load the FAISS index from the local path (with deserialization enabled)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Load the language model
    llm = load_llm()
    
    # Set up the custom prompt
    qa_prompt = set_custom_prompt()
    
    # Create a QA retrieval chain
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function to get the final result without source documents
def final_result(query):
    start_time = time.time()  # Record the start time
    qa_result = qa_bot()
    response = qa_result({'query': query})
    end_time = time.time()  # Record the end time
    
    # Calculate the response time
    response_time = end_time - start_time
    
    return response['result'], response_time  # Return both the response and the time taken

# Streamlit UI
def main():
    st.title("Medical Chatbot using LangChain and Streamlit")
    
    # Input box for user query
    query = st.text_input("Enter your medical question:")
    
    if st.button("Get Answer"):
        if query:
            # Call the model and fetch the result with the response time
            response, response_time = final_result(query)
            
            # Display the response
            st.write("Response:")
            st.write(response)
            
            # Display the response time
            st.write(f"Response generated in {response_time:.2f} seconds")

# Run the app
if __name__ == '__main__':
    main()
