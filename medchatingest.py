from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#to create file path for storing data and vector databases
Datapath="Data/"
DBfaisspath="vectorstores/db_faiss"

#for createing the vector database we are opening the pdfs and loading and chunking it directly
def create_vector_db():
    loader=DirectoryLoader(Datapath, glob="*.pdf", loader_cls=PyPDFLoader)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

    text=text_splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':'cpu'})

    db=FAISS.from_documents(text,embeddings)
    db.save_local(DBfaisspath)
    
if __name__=="__main__":
    create_vector_db()


    