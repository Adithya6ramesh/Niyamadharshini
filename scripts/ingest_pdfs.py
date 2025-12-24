"""
Offline PDF Ingestion Pipeline for Multilingual RAG
Complete end-to-end pipeline: Extract → Clean → Chunk → Embed → Store
Optimized for Malayalam and multilingual documents.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Import pipeline components
from ocr_extract import PDFTextExtractor
from clean_text import TextCleaner
from chunk_text import TextChunker
from embed_store import EmbeddingStore


class PDFIngestionPipeline:
    """Complete pipeline for PDF ingestion into RAG system."""
    
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 200,
        max_chunks: int = 40000,
        persist_directory: str = "chroma_db",
        collection_name: str = "malayalam_docs"
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            data_dir: Directory containing PDF files
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size to keep
            max_chunks: Maximum chunks allowed (safety limit)
            persist_directory: ChromaDB persistence directory
            collection_name: ChromaDB collection name
        """
        self.data_dir = data_dir
        self.max_chunks = max_chunks
        
        # Initialize pipeline components
        print("[INIT] Initializing pipeline components...")
        
        self.extractor = PDFTextExtractor(min_text_threshold=50)
        print("  ✓ PDFTextExtractor ready")
        
        self.cleaner = TextCleaner()
        print("  ✓ TextCleaner ready")
        
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        print("  ✓ TextChunker ready")
        
        # Embedding store initialized later (after we know we won't exceed limits)
        self.store = None
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        print("[INIT] Pipeline components ready\n")
    
    def find_pdfs(self) -> List[Path]:
        """
        Find all PDF files in data directory.
        
        Returns:
            List of PDF file paths
        """
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        pdf_files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.PDF"))
        
        if not pdf_files:
            print(f"[WARN] No PDF files found in {self.data_dir}")
            return []
        
        print(f"[INFO] Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def is_pdf_already_ingested(self, pdf_name: str) -> bool:
        """
        Check if a PDF is already ingested in ChromaDB.
        
        Args:
            pdf_name: Name of the PDF file
            
        Returns:
            True if PDF already exists in collection, False otherwise
        """
        try:
            from chromadb import PersistentClient
            client = PersistentClient(path=self.persist_directory)
            collection = client.get_collection(self.collection_name)
            results = collection.get(
                where={"source": pdf_name},
                limit=1
            )
            return len(results["ids"]) > 0
        except:
            return False
    
    def process_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Process a single PDF through extract → clean → chunk pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunk dictionaries
        """
        pdf_name = pdf_path.name
        
        print(f"\n{'='*70}")
        print(f"Processing: {pdf_name}")
        print(f"{'='*70}")
        
        # Step 1: Extract text
        print(f"\n[1/3] Extracting text...")
        pages = self.extractor.extract_from_pdf(str(pdf_path))
        num_pages = len(pages)
        print(f"  → Extracted {num_pages} pages")
        
        # Step 2: Clean text
        print(f"\n[2/3] Cleaning text...")
        cleaned_pages = self.cleaner.clean_pages(pages)
        
        # 🔥 NEW STEP: Merge small consecutive pages (OCR-safe)
        merged_pages = []
        buffer_text = ""
        buffer_pages = []

        for page in cleaned_pages:
            text = page["text"].strip()

            if len(text) < 200:
                buffer_text += text + "\n"
                buffer_pages.append(page["page"])
            else:
                if buffer_text:
                    merged_pages.append({
                        "page": buffer_pages,
                        "text": buffer_text.strip()
                    })
                    buffer_text = ""
                    buffer_pages = []

                merged_pages.append({
                    "page": page["page"],
                    "text": text
                })

        # Flush remaining buffer
        if buffer_text:
            merged_pages.append({
                "page": buffer_pages,
                "text": buffer_text.strip()
            })

        print(f"  → Merged {len(cleaned_pages)} pages into {len(merged_pages)} text blocks")
        
        # Step 3: Chunk text
        print(f"\n[3/3] Chunking text...")
        chunks = self.chunker.chunk_pages(merged_pages, pdf_name=pdf_name)
        print(f"  → Generated {len(chunks)} chunks")
        
        return chunks, num_pages
    
    def run_pipeline(self) -> Dict:
        """
        Run the complete ingestion pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        print("\n" + "="*70)
        print("STARTING PDF INGESTION PIPELINE")
        print("="*70)
        
        # Find PDFs
        pdf_files = self.find_pdfs()
        if not pdf_files:
            print("[ERROR] No PDFs to process")
            return {}
        
        # Process each PDF
        processed_pdfs_set = set()
        all_chunks = []
        total_pages = 0
        processed_pdfs = 0
        
        for pdf_path in pdf_files:
            if pdf_path.name in processed_pdfs_set:
                print(f"[SKIP] Already processed {pdf_path.name}")
                continue
            
            if self.is_pdf_already_ingested(pdf_path.name):
                print(f"[SKIP] {pdf_path.name} already embedded")
                continue

            try:
                # Process PDF
                chunks, num_pages = self.process_pdf(pdf_path)
                
                processed_pdfs_set.add(pdf_path.name)
                
                # Track statistics
                all_chunks.extend(chunks)
                total_pages += num_pages
                processed_pdfs += 1
                
                print(f"\n[PROGRESS] Total chunks so far: {len(all_chunks)}")
                
                # Safety check: Stop if exceeding chunk limit
                if len(all_chunks) > self.max_chunks:
                    print(f"\n[ERROR] Chunk limit exceeded!")
                    print(f"  Maximum allowed: {self.max_chunks}")
                    print(f"  Current count: {len(all_chunks)}")
                    print(f"  Processed {processed_pdfs}/{len(pdf_files)} PDFs")
                    print(f"\n[ACTION] Stopping pipeline to prevent memory issues")
                    sys.exit(1)
                
            except Exception as e:
                print(f"\n[ERROR] Failed to process {pdf_path.name}: {e}")
                continue
        
        # Print pre-embedding summary
        print("\n" + "="*70)
        print("PRE-EMBEDDING SUMMARY")
        print("="*70)
        print(f"Total PDFs processed: {processed_pdfs}")
        print(f"Total pages extracted: {total_pages}")
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Chunk limit check: {len(all_chunks)}/{self.max_chunks} ✓")
        print("="*70)
        
        # Step 4: Embed and store (only once at the end)
        if all_chunks:
            print(f"\n[EMBEDDING] Initializing embedding model...")
            self.store = EmbeddingStore(
                model_name="BAAI/bge-m3",
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            print(f"\n[EMBEDDING] Embedding {len(all_chunks)} chunks...")
            num_stored = self.store.embed_and_store(all_chunks, batch_size=32)
            
            # Print final summary
            print("\n" + "="*70)
            print("FINAL INGESTION SUMMARY")
            print("="*70)
            print(f"Total PDFs processed: {processed_pdfs}")
            print(f"Total pages: {total_pages}")
            print(f"Total chunks: {len(all_chunks)}")
            print(f"Total vectors stored: {num_stored}")
            print("="*70)
            
            # Print embedding store stats
            self.store.print_summary(show_example=True)
            
            return {
                "pdfs_processed": processed_pdfs,
                "total_pages": total_pages,
                "total_chunks": len(all_chunks),
                "vectors_stored": num_stored
            }
        else:
            print("\n[WARN] No chunks to embed")
            return {
                "pdfs_processed": processed_pdfs,
                "total_pages": total_pages,
                "total_chunks": 0,
                "vectors_stored": 0
            }


def main():
    """Main entry point for PDF ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into RAG system with Malayalam support"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing PDF files (default: data)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)"
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=200,
        help="Minimum chunk size to keep (default: 200)"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=40000,
        help="Maximum total chunks allowed (default: 40000)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="chroma_db",
        help="ChromaDB persistence directory (default: chroma_db)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="malayalam_docs",
        help="ChromaDB collection name (default: malayalam_docs)"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = PDFIngestionPipeline(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        max_chunks=args.max_chunks,
        persist_directory=args.db_path,
        collection_name=args.collection
    )
    
    # Run pipeline
    stats = pipeline.run_pipeline()
    
    # Exit with appropriate code
    if stats and stats.get("vectors_stored", 0) > 0:
        print("\n[SUCCESS] Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n[WARN] Pipeline completed with no vectors stored")
        sys.exit(1)


if __name__ == "__main__":
    main()
