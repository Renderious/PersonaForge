import streamlit as st
import os

from parser import chunk_text
from memory_db import store_chunks, query_chunks, get_or_create_collection
from llm_router import extract_characters, generate_character_system_prompt, chat_stream

# Page config
st.set_page_config(page_title="PersonaForge", layout="wide")
st.title("PersonaForge (Phase 1)")

# Initialize session state for storing app data
if "book_name" not in st.session_state:
    st.session_state.book_name = None
if "characters" not in st.session_state:
    st.session_state.characters = []
if "character_prompts" not in st.session_state:
    st.session_state.character_prompts = {}
if "selected_character" not in st.session_state:
    st.session_state.selected_character = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for book upload and character selection
with st.sidebar:
    st.header("1. Ingestion")

    uploaded_file = st.file_uploader("Upload a book (.txt)", type=["txt"])

    if uploaded_file is not None:
        # We only process if it's a new book
        current_book_name = uploaded_file.name.replace(".txt", "")

        if st.session_state.book_name != current_book_name:
            st.session_state.book_name = current_book_name
            st.session_state.characters = []
            st.session_state.character_prompts = {}
            st.session_state.selected_character = None
            st.session_state.messages = []

            with st.spinner(f"Processing '{current_book_name}'..."):
                text = uploaded_file.read().decode("utf-8")

                # 1. Parse and chunk
                st.write("Chunking text...")
                chunks = chunk_text(text)

                # 2. Store in ChromaDB
                st.write("Storing chunks in Vector DB...")
                # Ensure a clean collection if we're re-uploading the same named file
                collection_name = current_book_name
                try:
                    store_chunks(collection_name, chunks)
                except Exception as e:
                    st.error(f"Error storing chunks: {e}")

                # 3. Extract Characters
                st.write("Extracting characters with local LLM...")
                characters = extract_characters(chunks)
                st.session_state.characters = characters

                if not characters:
                    st.warning("No characters extracted.")

                st.success("Ingestion complete!")

    st.header("2. Character Selection")

    # If we have characters, display them
    if st.session_state.characters:
        character_choice = st.selectbox(
            "Select a character to chat with:",
            options=["-- Select --"] + st.session_state.characters
        )

        if character_choice != "-- Select --":
            # If a new character is selected, clear chat and generate prompt if needed
            if st.session_state.selected_character != character_choice:
                st.session_state.selected_character = character_choice
                st.session_state.messages = [] # Reset chat history

                # Generate system prompt if we haven't already
                if character_choice not in st.session_state.character_prompts:
                    with st.spinner(f"Generating persona for {character_choice}..."):
                        # Get some relevant chunks about the character for context
                        relevant_chunks = query_chunks(st.session_state.book_name, character_choice, n_results=5)

                        sys_prompt = generate_character_system_prompt(
                            character_name=character_choice,
                            text_chunks=relevant_chunks
                        )
                        st.session_state.character_prompts[character_choice] = sys_prompt

                st.success(f"Ready to chat with {character_choice}!")
    else:
        st.info("Upload a book to extract characters.")

# Main area for chatting
st.header("3. Chat Interface")

if st.session_state.selected_character:
    character = st.session_state.selected_character
    sys_prompt = st.session_state.character_prompts.get(character, "")

    with st.expander("View System Prompt (Developer Info)"):
        st.text_area("Core System Prompt", sys_prompt, height=150, disabled=True)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input(f"Message {character}..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant memories/chunks from ChromaDB
        with st.spinner("Recalling memories..."):
            collection_name = st.session_state.book_name
            context_chunks = query_chunks(collection_name, prompt, n_results=3)

        # Stream response from Ollama
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Use llm_router's stream function
            try:
                for response_chunk in chat_stream(
                    character_name=character,
                    system_prompt=sys_prompt,
                    context_chunks=context_chunks,
                    user_message=prompt
                ):
                    full_response += response_chunk
                    message_placeholder.markdown(full_response + "▌")

                # Final update without cursor
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error during chat: {e}")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload a book and select a character from the sidebar to start chatting.")
