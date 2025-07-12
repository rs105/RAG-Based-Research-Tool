# Import the Streamlit library for building the web UI
import streamlit as st

# Import your custom functions from rag.py
from rag import process_urls, generate_answer

# Title of the web app
st.title("Real Estate Research Tool")

# Sidebar inputs for user to enter urls
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

# Empty placeholder on the main page to display messages dynamically
placeholder = st.empty()

# Sidebar button to trigger URL processing
process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    # Collect all non-empty URLs entered by the user
    urls = [url for url in (url1, url2, url3) if url != '']
    if len(urls) == 0:
        # Warn the user if they didn't enter any valid URLs
        placeholder.text("You must provide at least one valid url")
    else:
        # Process each URL and display status updates using the placeholder - loading data, storing vectors to vector db...
        for status in process_urls(urls):
            placeholder.text(status)

# --- MAIN SECTION FOR QUESTION INPUT ---

# Use session state to keep track of the user's question
if "question" not in st.session_state:
    st.session_state.question = ""

# Input field for question
st.text_input("Question", key="question")

# Create Submit and Reset buttons
submit_button = st.button("Submit")

# Handle Submit button
if submit_button and st.session_state.question:
    try:
        # Call generate_answer to get the answer and sources
        answer, sources = generate_answer(st.session_state.question)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)

    except RuntimeError as e:
        placeholder.text("You must process urls first")

