"""Simple retrieval‑augmented chatbot.

This script loads a vector database and a SentenceTransformer embedding
model, retrieves the most relevant text chunks for a given question,
and then generates an answer using a generative language model.  By
default it uses OpenAI's chat completion API, but you can modify
`answer_question` to use a different model (e.g. Hugging Face models)
if you prefer or do not have access to the OpenAI API.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from .utils.helpers import get_embedding_model

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

# Try relative imports first; fall back to absolute imports when running as a script
try:
    from .utils.vector_db import VectorDatabase  # type: ignore
except ImportError:
    from app.utils.vector_db import VectorDatabase  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_config(config_module_name: str = "config") -> dict:
    """Load configuration module if present.

    This function attempts to import a module named `config` from the current
    working directory.  If the module is not found, an empty dict is returned.
    Any errors during import will propagate.
    """
    try:
        config_module = importlib.import_module(config_module_name)
        return {k: getattr(config_module, k) for k in dir(config_module) if k.isupper()}
    except ModuleNotFoundError:
        return {}


def build_prompt(contexts: List[str], question: str) -> str:
    """Construct the prompt fed to the language model."""
    context_text = "\n\n".join(contexts)
    prompt = (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the context is insufficient, say you don't know.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt


def answer_question(
    question: str,
    db: VectorDatabase,
    embed_model: SentenceTransformer,
    top_k: int = 5,
    openai_model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> str:
    """Retrieve context from the vector database and generate an answer.

    Args:
        question: User question string.
        db: Loaded vector database.
        embed_model: SentenceTransformer model used to encode the question.
        top_k: Number of context chunks to retrieve.
        openai_model: Name of the OpenAI ChatCompletion model to use.
        temperature: Sampling temperature for OpenAI model.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A string containing the generated answer.
    """
    if openai is None:
        raise RuntimeError(
            "OpenAI package is not installed. Install openai or modify answer_question to use another model."
        )
    # Encode the question
    query_embedding = embed_model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    # Retrieve top_k contexts
    results = db.search(query_embedding, top_k=top_k)
    if not results:
        return "I'm sorry, I could not find relevant information."
    context_texts = [db.metadata[idx]["text"] for idx, _ in results]
    prompt = build_prompt(context_texts, question)
    # Prepare messages for ChatCompletion
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who answers questions based on the provided context.",
        },
        {"role": "user", "content": prompt},
    ]
    # Read API key from environment or config
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try to load from config file
        config = load_config()
        api_key = config.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable or define it in config.py."
        )
    openai.api_key = api_key
    # Call OpenAI ChatCompletion
    try:
        response = openai.ChatCompletion.create(
            model=openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        answer = "I'm sorry, I encountered an error while generating the answer."
    return answer


def run_chat(args: argparse.Namespace) -> None:
    """CLI entry point for asking a question."""
    # Load vector database
    db = VectorDatabase.load(args.index_path, args.metadata_path)
    # Load embedding model
    embed_model = get_embedding_model(args.model_name)
    # Ask the question
    answer = answer_question(
        question=args.question,
        db=db,
        embed_model=embed_model,
        top_k=args.top_k,
        openai_model=args.openai_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(answer)