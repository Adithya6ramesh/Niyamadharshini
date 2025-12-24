"""
Multilingual RAG Embedding Pipeline
Embeds text chunks using BGE-M3 and stores in ChromaDB.
Supports Malayalam and other languages with persistent storage.
"""

import os
from typing import List, Dict
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm


class EmbeddingStore:
    """Generate embeddings and store in ChromaDB with persistence."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        persist_directory: str = "chroma_db",
        collection_name: str = "malayalam_docs"
    ):
        """
        Initialize the embedding store.
        
        Args:
            model_name: HuggingFace model name for embeddings
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Initialize ChromaDB with persistence
        print(f"[INFO] Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Malayalam and multilingual document embeddings"}
        )
        print(f"[INFO] Collection '{collection_name}' ready")
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of chunk dicts with 'text' key
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        texts = [chunk["text"] for chunk in chunks]
        
        # Normalize whitespace (collapse multiple spaces)
        texts = [" ".join(t.split()) for t in texts]
        
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"[INFO] Generated {len(embeddings)} embeddings")
        return embeddings.tolist()
    
    def store_embeddings(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Store embeddings and metadata in ChromaDB.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' keys
            embeddings: List of embedding vectors
            
        Returns:
            Number of vectors stored
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        
        # Check if collection already has data
        if self.collection.count() > 0:
            print(f"[WARN] Collection already has {self.collection.count()} vectors. Adding more...")
        
        print(f"[INFO] Storing {len(chunks)} vectors in ChromaDB...")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID
            metadata = chunk.get("metadata", {})
            chunk_id = metadata.get("chunk_id", i)
            page = metadata.get("page", 0)
            pdf_name = metadata.get("pdf_name", "unknown")
            
            # Create unique document ID
            doc_id = f"{pdf_name}_page{page}_chunk{chunk_id}"
            
            ids.append(doc_id)
            documents.append(chunk["text"])
            
            # Convert all metadata values to strings for ChromaDB compatibility
            safe_metadata = {
                "pdf_name": str(metadata.get("pdf_name", "unknown")),
                "page": str(metadata.get("page", 0)),
                "chunk_id": str(metadata.get("chunk_id", i)),
                "method": str(metadata.get("method", "unknown")),
                "char_count": str(metadata.get("char_count", len(chunk["text"]))),
                "source_char_count": str(metadata.get("source_char_count", 0))
            }
            
            metadatas.append(safe_metadata)
            embeddings_list.append(embedding)
        
        # Add to collection (ChromaDB handles batch inserts efficiently)
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"[INFO] Successfully stored {len(ids)} vectors")
        return len(ids)
    
    def embed_and_store(
        self,
        chunks: List[Dict],
        batch_size: int = 16
    ) -> int:
        """
        Complete pipeline: embed chunks and store in ChromaDB.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' keys
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of vectors stored
        """
        if not chunks:
            print("[WARN] No chunks to process")
            return 0
        
        print(f"[INFO] Embedding {len(chunks)} chunks...")
        
        # Extract texts
        texts = [c["text"] for c in chunks]
        
        # Normalize whitespace
        texts = [" ".join(t.split()) for t in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Prepare metadata
        metadatas = []
        for c in chunks:
            metadata = c.get("metadata", {})
            metadatas.append({
                "source": str(metadata.get("pdf_name", "unknown")),
                "page": str(metadata.get("page", 0)),
                "chunk_id": str(metadata.get("chunk_id", 0)),
                "method": str(metadata.get("method", "unknown"))
            })
        
        # Generate IDs
        ids = [
            f"{c['metadata'].get('pdf_name', 'unknown')}_p{c['metadata'].get('page', 0)}_{i}"
            for i, c in enumerate(chunks)
        ]
        
        # Check if collection already has data
        if self.collection.count() > 0:
            print(f"[WARN] Collection already has {self.collection.count()} vectors. Adding more...")
        
        # Store in ChromaDB
        print(f"[INFO] Storing {len(chunks)} vectors in ChromaDB...")
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[INFO] Successfully stored {len(ids)} vectors")
        return len(ids)
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the stored collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        stats = {
            "total_vectors": count,
            "collection_name": self.collection_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "persist_directory": self.persist_directory
        }
        
        return stats
    
    def print_summary(self, show_example: bool = True):
        """
        Print summary of stored embeddings.
        
        Args:
            show_example: Whether to show an example metadata entry
        """
        stats = self.get_collection_stats()
        
        print("\n" + "=" * 70)
        print("EMBEDDING STORAGE SUMMARY")
        print("=" * 70)
        print(f"Collection name: {stats['collection_name']}")
        print(f"Total vectors stored: {stats['total_vectors']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Persist directory: {stats['persist_directory']}")
        print("=" * 70)
        
        if show_example and stats['total_vectors'] > 0:
            # Retrieve one example
            print("\nEXAMPLE METADATA ENTRY:")
            print("-" * 70)
            
            results = self.collection.get(
                limit=1,
                include=["documents", "metadatas"]
            )
            
            if results and results['ids']:
                print(f"ID: {results['ids'][0]}")
                print(f"Metadata: {results['metadatas'][0]}")
                print(f"Document preview: {results['documents'][0][:200]}...")
            
            print("-" * 70)
    
    def query_similar(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict:
        """
        Query the collection for similar documents.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            
        Returns:
            Query results from ChromaDB
        """
        print(f"[INFO] Querying for: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text])[0].tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results


def main():
    """Demo usage of EmbeddingStore."""
    # Example chunks (Malayalam + English)
    sample_chunks = [
        {
            "text": "മലയാളം ഭാഷ വളരെ സുന്ദരമാണ്. ഇത് കേരളത്തിന്റെ ഔദ്യോഗിക ഭാഷയാണ്. മലയാളം ഭാഷയ്ക്ക് സമ്പന്നമായ സാഹിത്യ പാരമ്പര്യമുണ്ട്.",
            "metadata": {
                "pdf_name": "sample.pdf",
                "page": 1,
                "chunk_id": 0,
                "method": "pypdf",
                "char_count": 150,
                "source_char_count": 500
            }
        },
        {
            "text": "Malayalam is one of the 22 scheduled languages of India. It is spoken by over 38 million people worldwide. The language has a rich literary tradition.",
            "metadata": {
                "pdf_name": "sample.pdf",
                "page": 1,
                "chunk_id": 1,
                "method": "pypdf",
                "char_count": 152,
                "source_char_count": 500
            }
        },
        {
            "text": "കേരളം ദൈവത്തിന്റെ സ്വന്തം നാട് എന്നറിയപ്പെടുന്നു. പ്രകൃതി സൗന്ദര്യത്താൽ സമ്പന്നമായ ഈ നാട് ലോകമെമ്പാടും പ്രസിദ്ധമാണ്.",
            "metadata": {
                "pdf_name": "sample.pdf",
                "page": 2,
                "chunk_id": 2,
                "method": "ocr",
                "char_count": 135,
                "source_char_count": 400
            }
        }
    ]
    
    # Initialize embedding store
    store = EmbeddingStore(
        model_name="BAAI/bge-m3",
        persist_directory="chroma_db",
        collection_name="malayalam_docs"
    )
    
    # Embed and store chunks
    num_stored = store.embed_and_store(sample_chunks, batch_size=32)
    
    # Print summary
    store.print_summary(show_example=True)
    
    # Demo: Query similar documents
    print("\n" + "=" * 70)
    print("DEMO: SIMILARITY SEARCH")
    print("=" * 70)
    
    query = "What is Malayalam?"
    results = store.query_similar(query, n_results=2)
    
    print(f"\nTop results for query: '{query}'")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n[Result {i+1}] Distance: {distance:.4f}")
        print(f"Page: {metadata['page']}, Method: {metadata['method']}")
        print(f"Text: {doc[:200]}...")
    
    print("\n" + "=" * 70)
    
    return store


if __name__ == "__main__":
    main()
