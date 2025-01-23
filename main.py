import streamlit as st
import os
import re
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from tavily import TavilyClient
import tempfile

# Load environment variables
load_dotenv()

# Set API keys from environment
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
        search_result = client.search(
            query=query, 
            search_depth="advanced",
            include_images=True, 
            include_image_descriptions=True, 
            include_answer=True,
            max_results=7
        )
        return search_result
    except Exception as e:
        return f"Search error: {str(e)}"

def generate_comprehensive_response(prompt, doc_results, web_context):
    """
    Generate a comprehensive response integrating document and web search information
    with color-coded sections for better readability.
    """
    try:
        # Use a more advanced LLM for comprehensive synthesis
        llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o')
        
        # Comprehensive prompt with detailed instructions
        combined_prompt = (
            "You are an advanced AI assistant with access to both document knowledge and web search results. "
            "Provide a highly detailed, nuanced, and comprehensive answer. Follow these advanced guidelines:\n\n"
            "🔍 PRIMARY DOCUMENT KNOWLEDGE:\n"
            f"Detailed document context:\n{doc_results['answer']}\n"
            "- Extract comprehensive insights\n"
            "- Identify key theories and methodologies\n"
            "- Highlight potential implications\n\n"
            
            "🌐 WEB SEARCH CONTEXT:\n"
            f"Comprehensive web search findings:\n{web_context}\n"
            "- Perform advanced cross-referencing\n"
            "- Identify emerging trends\n"
            "- Address potential knowledge gaps\n\n"
            
            "🧠 SYNTHESIS REQUIREMENTS:\n"
            "- Create a multi-layered, intellectually rigorous response\n"
            "- Seamlessly integrate document and web search information\n"
            "- Maintain academic-level precision\n"
            "- Provide clear source attribution\n\n"
            
            f"ORIGINAL QUERY: {prompt}\n\n"
            "Response Guidelines:\n"
            "1. Begin with a comprehensive overview\n"
            "2. Provide in-depth analysis\n"
            "3. Use clear, structured formatting\n"
            "4. Highlight unique insights\n"
        )

        # Generate comprehensive response
        response = llm.predict(combined_prompt)

        # Add color-coded formatting for better readability
        formatted_response = (
            "## 📄 Document Knowledge\n"
            "```diff\n"
            "+ Insights from Uploaded Documents:\n"
            f"{doc_results['answer']}\n"
            "```\n\n"
            "## 🌐 Web Search Context\n"
            "```css\n"
            "/* Latest Web Search Findings */\n"
            f"{web_context}\n"
            "```\n\n"
            "## 🧠 Comprehensive Analysis\n"
            f"{response}"
        )

        return formatted_response

    except Exception as e:
        return f"Error generating comprehensive response: {str(e)}"

def generate_suggested_questions(response):
    """Generate suggested follow-up questions"""
    try:
        llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o-mini')
        suggested_prompt = f"Generate 3 concise follow-up questions based on this response:\n\n{response}"
        suggested_questions = llm.predict(suggested_prompt)
        return suggested_questions.strip().split('\n')
    except Exception:
        return [
            "Can you elaborate on that?",
            "Tell me more about this topic.",
            "What are the key takeaways?"
        ]

def initialize_system(uploaded_files):
    """Initialize system components with uploaded files"""
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

    # Create vector store using FAISS
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(document_chunks, embeddings)

    # Setup chat model
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o-mini')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        verbose=False,
        memory=memory
    )

    return llm, pdf_qa, vectordb, embeddings, document_splitter

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
                # Check if query is a web search or contains URLs
                is_web_search = prompt.startswith('/search')
                contains_url = URL_PATTERN.search(prompt)

                if is_web_search or contains_url:
                    # Remove /search prefix if present
                    clean_query = prompt[7:].strip() if is_web_search else prompt

                    # Web search or URL processing
                    web_sources = []
                    tavily_result = None

                    # Perform web search or URL processing
                    if is_web_search:
                        tavily_result = get_tavily_search(clean_query)
                        
                        # Process Tavily search results
                        if isinstance(tavily_result, dict):
                            # Add web search sources
                            web_sources.append(f"🌐 Web Search Query: {clean_query}")
                            
                            # Add Tavily answer if available
                            if tavily_result.get('answer'):
                                web_sources.append(f"📝 AI Generated Answer: {tavily_result['answer']}")
                            
                            # Add result sources
                            for result in tavily_result.get('results', []):
                                web_sources.append(f"🔗 Source: {result['title']} ({result['url']})")
                            
                            # Add images if available
                            if tavily_result.get('images'):
                                web_sources.append("🖼️ Related Images:")
                                for img in tavily_result['images']:
                                    web_sources.append(f"- {img.get('url', img) if isinstance(img, dict) else img}")

                    if contains_url:
                        urls = URL_PATTERN.finditer(clean_query)
                        for url_match in urls:
                            url = url_match.group()
                            documents = process_url_content(url)
                            if documents:
                                # Process URL documents
                                doc_chunks = st.session_state.document_splitter.split_documents(documents)
                                url_doc_store = FAISS.from_documents(doc_chunks, st.session_state.embeddings)
                                st.session_state.vectordb.merge_from(url_doc_store)
                                web_sources.append(f"📄 URL Source: {url}")

                    # Combine results
                    response_content = ""
                    if is_web_search and tavily_result and tavily_result.get('answer'):
                        response_content = tavily_result['answer']
                    else:
                        doc_results = st.session_state.pdf_qa({"question": clean_query})
                        response_content = doc_results['answer']

                    # Use comprehensive response generation
                    full_response = generate_comprehensive_response(
                        clean_query, 
                        {'answer': response_content}, 
                        "\n".join(web_sources)
                    )

                    # Display response
                    st.markdown(full_response)

                    # Generate and display suggested questions
                    suggested_questions = generate_suggested_questions(full_response)
                    st.markdown("**Suggested Follow-up Questions:**")
                    for q in suggested_questions:
                        st.markdown(f"- {q}")

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                else:
                    # Regular document-based query
                    doc_results = st.session_state.pdf_qa({"question": prompt})
                    response = doc_results['answer']

                    # Display response
                    st.markdown(response)

                    # Generate and display suggested questions
                    suggested_questions = generate_suggested_questions(response)
                    st.markdown("**Suggested Follow-up Questions:**")
                    for q in suggested_questions:
                        st.markdown(f"- {q}")

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
