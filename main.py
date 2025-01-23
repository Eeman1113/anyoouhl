import streamlit as st
import os
import re
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from tavily import TavilyClient
import tempfile

# Load environment variables
load_dotenv()

# Set API keys from environment (recommended for security)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

# URL detection regex pattern
URL_PATTERN = re.compile(
    r'(?:(?:https?|ftp|file)://|www\.|ftp\.)(?:\([-A-Z0-9+&@#/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#/%=~_|$?!:,.]*\)|[A-Z0-9+&@#/%=~_|$])',
    re.IGNORECASE
)

def download_pdf(url):
    """Download PDF from URL and save to temporary file"""
    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        return None
    except Exception as e:
        st.error(f"Error downloading PDF: {str(e)}")
        return None

def process_url_content(url):
    """Process content from URL based on file type"""
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()

    if path.endswith('.pdf'):
        pdf_path = download_pdf(url)
        if pdf_path:
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                os.unlink(pdf_path)  # Clean up temporary file
                return documents
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    return []

def get_tavily_search(query):
    """Get web search results using Tavily"""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        search_result = client.search(query=query, search_depth="advanced")
        return search_result
    except Exception as e:
        return f"Search error: {str(e)}"

def initialize_system(uploaded_files):
    """Initialize system components with uploaded files"""
    # Load and process documents
    documents = []
    try:
        for file in uploaded_files:
            file_ext = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name

            if file_ext == '.pdf':
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
            elif file_ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(temp_file_path)
                documents.extend(loader.load())
            elif file_ext == '.txt':
                loader = TextLoader(temp_file_path)
                documents.extend(loader.load())

            os.unlink(temp_file_path)  # Clean up temporary file
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

    # Split documents
    document_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
    document_chunks = document_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    os.makedirs("./data", exist_ok=True)
    vectordb = Chroma.from_documents(document_chunks, embedding=embeddings, persist_directory='./data')
    vectordb.persist()

    # Setup chat model
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        verbose=False,
        memory=memory
    )

    return llm, pdf_qa, vectordb, embeddings, document_splitter

def process_query_with_urls(query, llm, pdf_qa, vectordb, embeddings, document_splitter):
    """Process query containing URLs"""
    urls = URL_PATTERN.finditer(query)
    url_contents = []

    for url_match in urls:
        url = url_match.group()
        st.info(f"Processing URL: {url}")

        documents = process_url_content(url)
        if documents:
            doc_chunks = document_splitter.split_documents(documents)
            vectordb.add_documents(doc_chunks)
            url_contents.append(f"Content from {url} has been processed and added to the knowledge base.")
        else:
            search_results = get_tavily_search(f"site:{url}")
            url_contents.append(f"Web search results for {url}:\n{search_results}")

    clean_query = URL_PATTERN.sub('', query).strip()
    if not clean_query:
        clean_query = "Please summarize the content from the provided URLs"

    doc_results = pdf_qa({"question": clean_query})

    combined_prompt = (
        "You are an AI assistant with access to both document knowledge and web search results. "
        "Your task is to provide accurate, comprehensive answers by following these guidelines:\n\n"

        "1. DOCUMENT ANALYSIS:\n"
        f"Review this information from the document database:\n{doc_results['answer']}\n"
        "- Prioritize this information as your primary source\n"
        "- Extract key concepts and relevant details\n"
        "- Note specific examples and data points\n"
        "- Identify any limitations in the document information\n\n"

        "2. URL CONTENT ANALYSIS:\n"
        f"Additional context from processed URLs:\n{chr(10).join(url_contents)}\n"
        "- Use this to supplement document information\n"
        "- Cross-reference with document findings\n"
        "- Note any updates or new information\n"
        "- Highlight any contradictions or confirmations\n\n"

        "3. INTEGRATION REQUIREMENTS:\n"
        "- Begin with the most relevant document information\n"
        "- Supplement with URL content where appropriate\n"
        "- Clearly distinguish between document and URL sources\n"
        "- Address any contradictions between sources\n"
        "- Provide a balanced, comprehensive analysis\n\n"

        f"QUERY: {clean_query}\n\n"

        "Format your response to:\n"
        "1. Lead with the most relevant document-based information\n"
        "2. Integrate supporting URL content naturally\n"
        "3. Maintain clear source attribution\n"
        "4. Address all aspects of the query\n"
        "5. Highlight any important limitations or uncertainties\n"
    )

    return llm.predict(combined_prompt)

def main():
    st.set_page_config(page_title="Document Chat", page_icon="📄", layout="wide")
    st.title("📄 Document Chat Assistant")

    # Sidebar for file uploads
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF, DOCX, or TXT files", 
            type=['pdf', 'docx', 'doc', 'txt'], 
            accept_multiple_files=True
        )
        
        # Initialize button
        if st.button("Initialize Document Base"):
            if uploaded_files:
                st.session_state.llm, st.session_state.pdf_qa, \
                st.session_state.vectordb, st.session_state.embeddings, \
                st.session_state.document_splitter = initialize_system(uploaded_files)
                st.success("Document base initialized successfully!")
            else:
                st.warning("Please upload documents first.")

    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter your query"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query if system is initialized
        if 'pdf_qa' not in st.session_state:
            st.warning("Please initialize the document base first.")
            return

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # Check if query contains URLs
                if URL_PATTERN.search(prompt):
                    response = process_query_with_urls(
                        prompt, 
                        st.session_state.llm, 
                        st.session_state.pdf_qa, 
                        st.session_state.vectordb, 
                        st.session_state.embeddings, 
                        st.session_state.document_splitter
                    )
                else:
                    # Regular query processing
                    web_context = get_tavily_search(prompt)
                    doc_results = st.session_state.pdf_qa({"question": prompt})

                    combined_prompt = (
                        "You are an AI assistant with access to both document knowledge and web search results. "
                        "Your task is to provide accurate, comprehensive answers by following these guidelines:\n\n"

                        "1. PRIMARY DOCUMENT KNOWLEDGE:\n"
                        f"Review this information from the document database:\n{doc_results['answer']}\n"
                        "- This is your primary source of information\n"
                        "- Extract key concepts and relevant details\n"
                        "- Pay attention to specific examples and data points\n"
                        "- Note any limitations or gaps in the document information\n\n"

                        "2. WEB SEARCH CONTEXT:\n"
                        f"Consider these web search findings:\n{web_context}\n"
                        "- Use this to supplement document information\n"
                        "- Fill gaps in document knowledge\n"
                        "- Verify and cross-reference information\n"
                        "- Note any updates or new developments\n\n"

                        "3. INTEGRATION REQUIREMENTS:\n"
                        "- Start with document-based information\n"
                        "- Add relevant web search context\n"
                        "- Clearly indicate information sources\n"
                        "- Address any contradictions\n"
                        "- Ensure comprehensive coverage\n\n"

                        f"QUERY: {prompt}\n\n"

                        "Format your response to:\n"
                        "1. Lead with the most relevant document-based information\n"
                        "2. Integrate web search findings naturally\n"
                        "3. Maintain clear source attribution\n"
                        "4. Address all aspects of the query\n"
                        "5. Highlight any important limitations or uncertainties\n"
                    )

                    response = st.session_state.llm.predict(combined_prompt)

                # Display assistant response
                st.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
