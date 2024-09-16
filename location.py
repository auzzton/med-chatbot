
# Import  libraries
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (Google Gemini API Key, etc.)
load_dotenv()

# Streamlit app for interactive input/output
def main():
    st.title("Hospital Finder")

    # Load the PDF file
    st.write("Loading  Data...")
    loader = PyPDFLoader("all india hospital list.pdf")
    data = loader.load()
    st.write("Data loaded successfully!")

    # Split the PDF into smaller chunks (for better retrieval)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    st.write(f"Total number of data chunks: {len(docs)}")

    # Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a vector store using Chroma
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    # Set up the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize Google Generative AI LLM for answering questions
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

    # Create a prompt for the question-answer task
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )

    # Define the prompt template for system and human inputs
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Create a question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Take user input as a question
    question = st.text_input("Enter your question about the document:")
    
    if question:
        # Retrieve relevant context from the PDF and generate the answer
        response = rag_chain.invoke({"input": question})
        st.write("Answer:")
        st.write(response.get("answer", "No relevant answer found."))

if __name__ == '__main__':
    main()
