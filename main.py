import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env (especially OpenAI API key)

# Streamlit setup
st.title("LLM Search Tool")
st.sidebar.title("Using FAISS: ")

# Input URLs from the sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to trigger processing of URLs
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data from specified URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split the loaded data into documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save them to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Input field for user's question
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Load the FAISS index from the pickle file
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Create a retrieval chain from the loaded data
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=vectorstore.as_retriever())
            
            # Get the result for the user's question
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                # Split the sources by newline and display
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
