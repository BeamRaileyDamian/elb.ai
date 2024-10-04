import os
import argparse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import *
from embedder import get_embedding_function

def create_retriever():
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

def setup():
    load_dotenv()
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key: raise ValueError("API key for Groq is not set in environment variables")
    return create_retriever(), groq_api_key

def create_template():
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context. The following data should be treated as facts you are familiar with:

    {context}

    ---

    Answer the question based on the above context without mentioning that these were provided to you just now: {question}.
    """
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs(results):
    return "\n\n".join(doc.page_content for doc in results)

def create_model(groq_api_key):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name='llama-3.1-8b-instant',
        temperature=0
    )

def rag_pipeline(query_text, retriever, groq_api_key):
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough(),
        }
        | create_template()
        | create_model(groq_api_key)
        | StrOutputParser()
    )
    response = rag_chain.invoke(query_text)
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    retriever, groq_api_key = setup()
    rag_pipeline(query_text, retriever, groq_api_key)

if __name__ == "__main__":
    main()