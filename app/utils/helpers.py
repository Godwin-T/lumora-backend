"""Generate vector embeddings for text chunks using Sentence Transformers.

This script reads a JSON file containing preprocessed chunks (as produced by
`preprocess_text.py`), encodes each chunk into a dense vector using a
pre‑trained SentenceTransformer model, and writes the resulting matrix of
embeddings to a NumPy `.npy` file.  It also writes out a metadata file
containing information about each chunk (source file, chunk index, and the
original text), so that you can map back from vector indices to the
underlying text during retrieval.
"""

from __future__ import annotations

import logging
import os
import re
import pickle
from pathlib import Path
import pdfplumber
from typing import Any, Dict, List, Iterable

import time
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


import nltk

# Global variables for caching the database and its last load time
cache = {}
last_load_time = 0
cache_lifetime = 24 * 60 * 60  # 24 hours in seconds
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and return a sentence transformer model.

    Args:
        model_name: Name or path of a SentenceTransformer model.

    Returns:
        A loaded SentenceTransformer model.
    """
    global last_load_time, cache
    
    current_time = time.time()
    if model_name in cache and (current_time - last_load_time) < cache_lifetime:
        print(f"Using cached model for {model_name}")
        return cache[model_name]

    logging.info(f"Loading embedding model '{model_name}'")
    model = SentenceTransformer(model_name)
    cache[model_name] = model
    last_load_time = current_time    
    return model


def embed_documents(
    model: SentenceTransformer, texts: List[str], batch_size: int = 32
) -> np.ndarray:
    """Embed a list of texts using the given model.

    Args:
        model: A SentenceTransformer model instance.
        texts: List of text strings to encode.
        batch_size: Number of texts to encode per batch.

    Returns:
        A NumPy array of shape (len(texts), embedding_dim) containing the embeddings.
    """
    logging.info(f"Encoding {len(texts)} texts in batches of {batch_size}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings

"""Utilities for extracting text from PDF files.

This module uses `pdfplumber` to read PDF documents and return their textual
content.  You can import the functions into your own code or run the script
directly from the command line.
"""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a single PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A string containing the concatenated text from all pages.
    """
    logging.debug(f"Opening PDF file: {pdf_path}")
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logging.warning(
                    f"Failed to extract text from page {page_number} of {pdf_path}: {e}"
                )
                page_text = ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_texts_from_directory(pdf_dir: str) -> Dict[str, str]:
    """Extract text from all PDF files in a directory.

    Args:
        pdf_dir: Path to the directory containing PDF files.

    Returns:
        A dictionary mapping each PDF file name to its extracted text.
    """
    pdf_dir_path = Path(pdf_dir)
    texts: Dict[str, str] = {}
    for pdf_file in pdf_dir_path.glob("*.pdf"):
        logging.info(f"Extracting text from {pdf_file.name}")
        texts[pdf_file.name] = extract_text_from_pdf(str(pdf_file))
    return texts

"""Text preprocessing utilities.

This module provides functions to clean raw text and split it into
overlapping chunks suitable for embedding.  The splitting strategy
operates on sentences to avoid cutting sentences in the middle.  You can
run this module directly from the command line to preprocess a JSON file
containing extracted PDF texts.
"""

def clean_text(text: str) -> str:
    """Normalize whitespace and remove extra line breaks.

    Args:
        text: Raw text extracted from a PDF.

    Returns:
        A cleaned version of the text.
    """
    # Replace carriage returns and tabs with spaces
    text = text.replace("\r", " ").replace("\t", " ")
    # Normalize newlines to spaces
    text = text.replace("\n", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split cleaned text into sentences using NLTK's Punkt tokenizer.

    Args:
        text: Cleaned text.

    Returns:
        A list of sentences.
    """
    # Ensure the tokenizer is available
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    return nltk.sent_tokenize(text)


def chunk_sentences(
    sentences: Iterable[str], chunk_size: int = 500, overlap: int = 50
) -> List[str]:
    """Group sentences into overlapping text chunks.

    The function accumulates sentences until the approximate token count
    reaches `chunk_size`.  Tokens are approximated by splitting on
    whitespace.  Each chunk overlaps with the previous one by `overlap`
    tokens to provide context across boundaries.

    Args:
        sentences: An iterable of sentence strings.
        chunk_size: Target number of tokens per chunk.
        overlap: Number of tokens of overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    def add_current_chunk():
        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

    for sentence in sentences:
        words = sentence.split()
        sentence_len = len(words)
        # If adding this sentence exceeds chunk size, finalize the current chunk
        if current_tokens + sentence_len > chunk_size and current_tokens > 0:
            add_current_chunk()
            # Create new chunk with overlap from previous chunk
            if overlap > 0 and chunks:
                # Take last `overlap` tokens from the previous chunk
                last_chunk_words = chunks[-1].split()
                overlap_tokens = last_chunk_words[-overlap:] if len(last_chunk_words) >= overlap else last_chunk_words
                current_chunk[:] = [" ".join(overlap_tokens)]
                current_tokens = len(overlap_tokens)
            else:
                current_chunk[:] = []
                current_tokens = 0
        current_chunk.append(sentence)
        current_tokens += sentence_len

    # Append any remaining chunk
    add_current_chunk()
    return chunks


def preprocess_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Clean text and split into chunks.

    Args:
        text: Raw text string.
        chunk_size: Target number of tokens per chunk.
        overlap: Number of overlapping tokens between chunks.

    Returns:
        A list of cleaned text chunks.
    """
    cleaned = clean_text(text)
    sentences = split_into_sentences(cleaned)
    return chunk_sentences(sentences, chunk_size=chunk_size, overlap=overlap)
