import os
import sys
import time
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from retriever import rag_pipeline, setup

def main():
    if 'retriever' not in st.session_state or 'groq_api_key' not in st.session_state:
        st.session_state.retriever, st.session_state.groq_api_key = setup()

    st.title(":robot_face: elb.ai :robot_face:")
    st.sidebar.markdown("### Sample Questions:")
    st.sidebar.markdown("- **What to do if i get an INC?**")
    st.sidebar.markdown("- **How can I get my UP ID??**")
    st.sidebar.markdown("- **Can i drop courses??** ")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        response = rag_pipeline(prompt, st.session_state.retriever, st.session_state.groq_api_key)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            messagePlaceholder = st.empty()
            typedResponse = ""
            if response:
                for char in response: # added typing effect
                    typedResponse += char
                    messagePlaceholder.markdown(typedResponse)
                    time.sleep(0.01)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()