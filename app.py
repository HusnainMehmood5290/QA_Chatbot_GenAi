import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
import tempfile
import os

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Reply: ", message.content)

def save_pdfs_to_temp(pdf_docs):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()    
    # List to store paths of saved PDFs
    pdf_paths = []
    for pdf in pdf_docs:
        # Define the file path for each PDF
        file_path = os.path.join(temp_dir, pdf.name)
        # Save the PDF to the temporary directory
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
        # Append the full path to the list
        pdf_paths.append(file_path)  # Add the file path directly
    return pdf_paths  # Return a list of file paths

def main():
    st.set_page_config("Information Retrieval")
    st.header("Information Retrieval System💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                created_temp_file = save_pdfs_to_temp(pdf_docs)
                raw_text = get_pdf_text(created_temp_file)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                
                # Properly initialize the conversation chain
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")

if __name__ == "__main__":
    main()
