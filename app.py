import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader, PlaywrightURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Import updated OpenAI classes
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()

# --- UI Configuration ---
st.set_page_config(page_title="RockyBot: News Research Tool", page_icon="📈")
st.title("RockyBot: News Research Tool 📈")

# --- Sidebar ---
st.sidebar.title("News Article URLs")
urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}") for i in range(3)]
loader_option = st.sidebar.selectbox(
    "Document Loader",
    ["UnstructuredURLLoader", "SeleniumURLLoader", "PlaywrightURLLoader"],
)

# --- State Management ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Process Button Logic ---
if st.sidebar.button("Process URLs"):
    with st.spinner("Processing URLs..."):
        try:
            # Dynamic Loader Selection
            if loader_option == "UnstructuredURLLoader":
                loader = UnstructuredURLLoader(urls=urls)
            elif loader_option == "SeleniumURLLoader":
                loader = SeleniumURLLoader(urls=urls)
            else:  # PlaywrightURLLoader
                loader = PlaywrightURLLoader(urls=urls)

            data = loader.load()

            # Optimized Text Splitting
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(data)

            # Embedding and Vector Store Creation
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state.vectorstore = vectorstore

            st.success("Data processing complete!")
        except Exception as e:
            st.error(f"An error occurred while processing URLs: {e}")

# --- Query Input and Response ---
query = st.text_input("Question:")

if query and st.session_state.vectorstore:
    with st.spinner("Searching for answers..."):
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=OpenAI(temperature=0.9),
            retriever=st.session_state.vectorstore.as_retriever(),
        )

        # Use invoke for chain execution
        result = chain.invoke({"question": query})

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
