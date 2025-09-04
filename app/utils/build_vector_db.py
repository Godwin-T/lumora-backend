#!/usr/bin/env python3
import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from app.utils.vector_db import run_build
from app.utils.helpers import (
    get_embedding_model, 
    extract_text_from_pdf, 
    preprocess_text, 
    embed_documents)


def process_pdf(pdf_path, extracted_text_path):
    """Extract text from PDF and save to JSON file"""
    pdf_extract = extract_text_from_pdf(pdf_path)
    out_path = Path(extracted_text_path)
    out_path.write_text(json.dumps({Path(pdf_path).name: pdf_extract}, ensure_ascii=False, indent=2))
    return out_path


def preprocess_extracted_text(extracted_text_path, preprocessed_text_path):
    """Preprocess extracted text and save chunks"""
    data = json.loads(Path(extracted_text_path).read_text())
    
    output_data = {}
    for filename, raw_text in data.items():
        chunks = preprocess_text(raw_text)
        output_data[filename] = chunks
    
    out_path = Path(preprocessed_text_path)
    out_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
    return out_path


def create_embeddings(preprocessed_text_path, embedding_path, metadata_path):
    """Create embeddings and metadata from preprocessed text"""
    embedding_model = get_embedding_model()
    data = json.loads(Path(preprocessed_text_path).read_text())
    
    # Flatten all chunks into a single list and record metadata
    texts = []
    metadata = []
    for filename, chunks in data.items():
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({"filename": filename, "chunk_id": idx, "text": chunk})
    
    embeddings = embed_documents(embedding_model, texts)
    embeddings_path = Path(embedding_path)
    np.save(embeddings_path, embeddings)
    
    metadata_path = Path(metadata_path)
    with metadata_path.open("wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata with {len(metadata)} entries to {metadata_path}")
    
    return embeddings_path, metadata_path


def build_vector_database(
    pdf_path,
    extracted_text_path,
    preprocessed_text_path,
    embedding_path,
    metadata_path,
    index_out
):
    """Build a vector database from a PDF document"""
    # Step 1: Extract text from PDF
    extracted_path = process_pdf(pdf_path, extracted_text_path)
    
    # Step 2: Preprocess the extracted text
    preprocessed_path = preprocess_extracted_text(extracted_path, preprocessed_text_path)
    
    # Step 3: Create embeddings and metadata
    embeddings_path, metadata_path = create_embeddings(
        preprocessed_path, embedding_path, metadata_path
    )
    
    # Step 4: Build the vector database
    run_build(embedding_path, index_out, metadata_out=metadata_path)
    print(f"Vector database built successfully and saved to {index_out}")


def main():
    parser = argparse.ArgumentParser(description="Build a vector database from a PDF document")
    
    parser.add_argument(
        "--pdf-path", "-p",
        type=str,
        default="./data/raw/Business registration knowledge base.pdf",
        help="Path to the PDF file"
    )
    
    parser.add_argument(
        "--extracted-text-path",
        type=str,
        default="./data/processed/extracted_text.json",
        help="Path to save the extracted text JSON"
    )
    
    parser.add_argument(
        "--preprocessed-text-path",
        type=str,
        default="./data/processed/preprocessed_text.json",
        help="Path to save the preprocessed text JSON"
    )
    
    parser.add_argument(
        "--embedding-path",
        type=str,
        default="./data/processed/embeddings.npy",
        help="Path to save the embeddings"
    )
    
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="./data/processed/metadata.pkl",
        help="Path to save the metadata"
    )
    
    parser.add_argument(
        "--index-out",
        type=str,
        default="./data/processed/index.faiss",
        help="Path to save the FAISS index"
    )
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    for path in [args.extracted_text_path, args.preprocessed_text_path, 
                args.embedding_path, args.metadata_path, args.index_out]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    build_vector_database(
        args.pdf_path,
        args.extracted_text_path,
        args.preprocessed_text_path,
        args.embedding_path,
        args.metadata_path,
        args.index_out
    )


if __name__ == "__main__":
    main()
