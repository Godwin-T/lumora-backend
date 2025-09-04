# RAG PDF Chatbot Project

This project provides a simple starting point for building a Retrieval‑Augmented Generation (RAG) system from a collection of PDF documents.  The goal is to extract text from PDFs, preprocess it, generate vector embeddings, store those vectors in a searchable database, and finally retrieve relevant passages to augment a language model’s answers.

## Project structure

```
rag_project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── extract_pdf_text.py       # Extracts text from PDF files
├── preprocess_text.py        # Cleans and splits text into chunks
├── generate_embeddings.py    # Generates vector embeddings for chunks
├── vector_db.py              # FAISS‑based vector database utilities
├── chatbot.py                # Retrieval and answer generation pipeline
└── config_example.py         # Example configuration file for API keys
```

## Setup

1. **Install dependencies**

   Make sure you are using Python 3.8 or later.  Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   If you plan to use a GPU version of FAISS or Sentence Transformers, adjust the dependencies accordingly.

2. **Prepare your PDFs**

   Place your PDF files in a directory (e.g. `./data/pdfs`).  The scripts expect to be given paths to these files when run.

3. **Extract and preprocess text**

   Use `extract_pdf_text.py` to extract raw text from each PDF.  Then, use `preprocess_text.py` to clean the text and split it into overlapping chunks suitable for embedding.

4. **Generate embeddings and build a vector database**

   Run `generate_embeddings.py` to create embeddings for your text chunks.  Then, use `vector_db.py` to build a FAISS index for similarity search.  The index and associated metadata can be saved to disk for later reuse.

5. **Query and answer questions**

   `chatbot.py` demonstrates a simple retrieval‑augmented pipeline.  It encodes a user question into a vector, retrieves the most relevant document chunks from the vector database, and then uses a language model to generate a final answer.  The default implementation includes a skeleton for using OpenAI’s API—remember to supply your API key (see `config_example.py`).  You can substitute a local generative model by modifying the `answer_question` function.

## Usage examples

### Extract text from PDFs

```bash
python extract_pdf_text.py --pdf-dir data/pdfs --output data/raw_texts.json
```

### Preprocess text and split into chunks

```bash
python preprocess_text.py --input data/raw_texts.json --output data/chunks.json --chunk-size 500 --overlap 50
```

### Generate embeddings

```bash
python generate_embeddings.py --input data/chunks.json --output-embeddings data/embeddings.npy --output-metadata data/metadata.pkl --model-name all-MiniLM-L6-v2
```

### Build the vector database

```bash
python vector_db.py build --embeddings data/embeddings.npy --metadata data/metadata.pkl --index-out data/index.faiss
```

### Ask a question

Before running the chatbot, copy `config_example.py` to `config.py` and set your OpenAI API key:

```bash
cp config_example.py config.py
# edit config.py and set OPENAI_API_KEY = "sk-..."

python chatbot.py --index data/index.faiss --metadata data/metadata.pkl --model-name all-MiniLM-L6-v2 --question "What does the document say about X?"
```

The script will retrieve the most relevant chunks and then use OpenAI’s API to generate an answer.  If you do not wish to use OpenAI, feel free to modify `answer_question` to integrate another model (e.g. via Hugging Face’s `transformers` library).

## Notes

* The default embedding model used here is the light‑weight `all-MiniLM-L6-v2`, which provides a good balance between performance and accuracy.  You can choose another SentenceTransformer model by passing a different name to the scripts.
* `vector_db.py` uses FAISS’s `IndexFlatL2` for exact nearest‑neighbor search.  For very large datasets you may want to explore approximate indexes (e.g. `IndexIVFFlat` or `IndexHNSWFlat`) to speed up queries.
* The project uses `nltk` for sentence tokenization.  Ensure that the Punkt tokenizer is downloaded the first time you run the script; if not, run the following once in Python:

  ```python
  import nltk
  nltk.download('punkt')
  ```

* See `config_example.py` for how to store your API keys and other settings.
