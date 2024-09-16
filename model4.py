import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import google.generativeai as genai
from pathlib import Path
from googletrans import Translator

# Patch for httpcore error
import httpcore
setattr(httpcore, 'SyncHTTPTransport', any)

# Configure GenAI API key
genai.configure(api_key="google api key")

# Path to your FAISS index
DB_FAISS_PATH = 'vectorstores/db_faiss'

# Custom prompt templates
qa_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Provide a detailed and helpful answer below:
Helpful answer:
"""

vision_ai_prompt_template = """
Analyze the medical image provided, focusing on detecting skin diseases or abnormalities in X-rays. 
Consider the following medical details when providing your answer:

1. For skin diseases, look for texture irregularities, color changes, or any signs of infection or malignancy (e.g., moles, rashes, etc.).
2. For X-rays, identify fractures, dislocations, or any visible abnormalities in bones or soft tissues.

Provide a detailed analysis based on the image:
Context: {context}
Medical Image Analysis:
"""

# Function to initialize the GenAI model
def initialize_genai_model():
    generation_config = {"temperature": 0.9}
    return genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Function to process the image and generate content based on medical prompts
def generate_medical_content(model, image_path, prompts):
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_path.read_bytes()
    }
    
    results = []
    for prompt_text in prompts:
        prompt_parts = [vision_ai_prompt_template.format(context=prompt_text), image_part]
        response = model.generate_content(prompt_parts)
        
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_part = candidate.content.parts[0]
                if text_part.text:
                    results.append(f"Prompt: {prompt_text}\nMedical Description:\n{text_part.text}\n")
                else:
                    results.append(f"Prompt: {prompt_text}\nMedical Description: No valid content generated.\n")
            else:
                results.append(f"Prompt: {prompt_text}\nMedical Description: No content parts found.\n")
        else:
            results.append(f"Prompt: {prompt_text}\nMedical Description: No candidates found.\n")
    
    return results

# Function to translate text into selected language
def translate_text(text, lang):
    translator = Translator()
    translation = translator.translate(text, dest=lang)
    return translation.text

# Setup custom prompt for QA bot
def set_custom_prompt():
    prompt = PromptTemplate(template=qa_prompt_template, input_variables=['context', 'question'])
    return prompt

# Limit the length of the input to avoid token limit issues
def truncate_text(text, max_tokens=512):
    tokens = text.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    return text

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 1}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Load the language model (CTransformers with Llama)
def load_llm():
    llm = CTransformers(
        model="D:/python medical chatbot/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=128,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Output function to get the final result without source documents
def final_result(query):
    start_time = time.time()
    query = truncate_text(query, max_tokens=400)
    qa_result = qa_bot()
    response = qa_result({'query': query})
    end_time = time.time()
    response_time = end_time - start_time

    if "cancer" in query.lower():
        danger_level = "high"
    elif "flu" in query.lower():
        danger_level = "moderate"
    else:
        danger_level = "low"
    
    return response['result'], response_time, danger_level

# Function to set the text color based on danger level
def set_text_color(text, danger_level):
    if danger_level == "high":
        color = "red"
        outline = "2px solid red"
    elif danger_level == "moderate":
        color = "orange"
        outline = "2px solid orange"
    else:
        color = "yellow"
        outline = "2px solid yellow"
    
    st.markdown(
        f"""
        <div style="border:{outline}; padding: 10px; color:{color};">
            {text}
        </div>
        """, unsafe_allow_html=True
    )

# Home page content
# Home page content with updated color
def show_home():
    st.title("Welcome to the CareAI")
    st.markdown("""<style>
                    .main {background-color: #201E43;}
                    .stApp {color: #F0DE36;}  /* Changed text color to F0DE36 */
                   </style>""", unsafe_allow_html=True)
    st.markdown("""
        **AN AI THAT CARES FOR YOU**  :
        1. **Medical Chatbot**: Ask medical-related questions and receive accurate answers from our AI.
        2. **Medical Image Interpreter**: Upload images (like X-rays or skin conditions), and the AI will provide an analysis based on medical prompts.
    """)

    # Brief explaining how to use the prompts
    st.markdown("""
        ### How to Use the CareAI :

        #### 1. **CareAI Chatbot**:
        - You can ask the chatbot medical-related questions, such as symptoms, diseases, or general health queries.
        - Example Questions:
          - *"What are the symptoms of flu?"*
          - *"What is the treatment for lung cancer?"*
        - The AI will provide detailed responses based on the context of your question. It will also evaluate the severity of the disease (e.g., **High**, **Moderate**, or **Low** severity) based on the input.

        #### 2. **Medical Image Interpreter**:
        - You can upload medical images, such as X-rays or skin condition photos, and receive a detailed analysis.
        - To get the best results, provide specific prompts to guide the AI in what to look for in the image.
        - Example Prompts:
          - *"Analyze skin condition"*
          - *"Detect fractures in X-ray"*
        - After uploading the image, enter one or more prompts in the input box (one prompt per line). The AI will process the image and provide descriptions based on your prompts.

        #### 3. **History**:
        - This section will show all the past questions and images you've analyzed, along with the responses provided by the AI.
    """)


# Streamlit app for both functionalities with a home page
def main():
    # Initialize session state for prompts and results
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "prompts" not in st.session_state:
        st.session_state.prompts = ""
    if "results" not in st.session_state:
        st.session_state.results = []
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "page" not in st.session_state:
        st.session_state.page = "Home"  # Set default page to Home

    # Sidebar for navigation (moving the navigation bar to the sidebar)
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("", ["Home", "Medical Chatbot", "Medical Image Interpreter", "History"])
    st.session_state.page = selected_page

    # Navigate to the appropriate page
    if st.session_state.page == "Home":
        show_home()
    
    elif st.session_state.page == "Medical Chatbot":
        st.title("Medical Chatbot")
        query = st.text_input("Enter your medical question:")
        if st.button("Send"):
            if query:
                response, response_time, danger_level = final_result(query)
                st.session_state.chat_history.append((query, response, danger_level))
        if st.session_state.chat_history:
            for question, response, danger_level in st.session_state.chat_history:
                st.write(f"**You**: {question}")
                set_text_color(f"**Bot**: {response}", danger_level)
                st.write(f"Disease severity: {danger_level.capitalize()}")
                st.write("---")

    elif st.session_state.page == "Medical Image Interpreter":
        st.title("Medical Image Interpreter")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            # Save the uploaded file
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize the GenAI model
            model = initialize_genai_model()
            
            # Input for multiple medical prompts
            st.write("Enter medical prompts (one per line, e.g., 'Analyze skin condition' or 'Detect fractures in X-ray'):")
            st.session_state.prompts = st.text_area("Medical Prompts", value=st.session_state.prompts)
            
            # Button to generate medical description
            if st.button("Generate Medical Description"):
                prompts = [prompt.strip() for prompt in st.session_state.prompts.split('\n') if prompt.strip()]
                
                if prompts:
                    image_path = Path("temp_image.jpg")
                    st.session_state.results = generate_medical_content(model, image_path, prompts)
                    st.session_state.history.append({
                        "image": uploaded_file,
                        "results": st.session_state.results
                    })
                else:
                    st.write("Please enter at least one prompt.")
            
            Path("temp_image.jpg").unlink()
        
        if st.session_state.uploaded_file and st.session_state.results:
            st.image(st.session_state.uploaded_file, caption='Uploaded Medical Image.', use_column_width=True)
            st.write("(Medical Image Interpreter):")
            for description in st.session_state.results:
                st.write(description)

    elif st.session_state.page == "History":
        st.title("History")
        if st.session_state.history:
            for i, entry in enumerate(st.session_state.history):
                st.image(entry['image'], caption=f'Uploaded Image {i+1}', use_column_width=True)
                for description in entry['results']:
                    st.write(description)
        else:
            st.write("No history available.")

# Run the app
if __name__ == '__main__':
    main()
