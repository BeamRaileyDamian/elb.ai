# elb.ai

## Installation
- Create and activate a virtual environment

```
python -m venv venv
venv\Scripts\activate
```

- Install dependencies

```
pip install -r requirements.txt
```

- Create .env file and supply values

## Running

- Embedder

Used for adding the documents inside data/ to the RAG system. The embeddings will be saved under chroma/.

```
python src\embedder.py  
```

- Chatbot
```
streamlit run ui\elb.ai.py   
```
