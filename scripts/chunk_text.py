"""
Malayalam-Safe Text Chunking for RAG
Uses LangChain RecursiveCharacterTextSplitter to create semantic chunks.
Preserves Malayalam characters and attaches comprehensive metadata.
"""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    """Chunk cleaned text for RAG with Malayalam preservation."""
    
    def __init__(
          self,
          chunk_size: int = 800,
          chunk_overlap: int = 100,
          min_chunk_size: int = 100
):

        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size to keep (filter smaller chunks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Initialize LangChain text splitter with Malayalam-safe separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
            is_separator_regex=False
        )
    
    def chunk_pages(
        self,
        pages: List[Dict],
        pdf_name: str = "document.pdf"
    ) -> List[Dict]:
        """
        Chunk a list of page dictionaries into smaller text chunks.
        
        Args:
            pages: List of dicts with keys: page, text, char_count, method (optional)
            pdf_name: Name of the source PDF file
            
        Returns:
            List of chunk dicts with metadata: {
                "text": str,
                "metadata": {
                    "pdf_name": str,
                    "page": int,
                    "chunk_id": int,
                    "method": str,
                    "char_count": int,
                    "source_char_count": int
                }
            }
        """
        all_chunks = []
        global_chunk_id = 0
        
        for page_data in pages:
            page_num = page_data.get("page", 0)
            page_text = page_data.get("text", "")
            extraction_method = page_data.get("method", "unknown")
            source_char_count = page_data.get("char_count", len(page_text))
            
            # Skip empty pages
            if not page_text or len(page_text.strip()) == 0:
                print(f"[INFO] Skipping empty page {page_num}")
                continue
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(page_text)
            
            # Process each chunk
            for chunk_text in chunks:
                # Filter out chunks that are too small
                if len(chunk_text.strip()) < self.min_chunk_size:
                    print(
                        f"[WARN] Skipping small chunk on page {page_num} "
                        f"({len(chunk_text)} chars < {self.min_chunk_size})"
                    )
                    continue
                
                # Create chunk with metadata
                chunk_dict = {
                    "text": chunk_text,
                    "metadata": {
                        "pdf_name": pdf_name,
                        "page": page_num,
                        "chunk_id": global_chunk_id,
                        "method": extraction_method,
                        "char_count": len(chunk_text),
                        "source_char_count": source_char_count
                    }
                }
                
                all_chunks.append(chunk_dict)
                global_chunk_id += 1
        
        return all_chunks
    
    def chunk_document(
        self,
        pages: List[Dict],
        pdf_name: str = "document.pdf"
    ) -> List[Dict]:
        """
        Chunk entire document (all pages combined) into chunks.
        Alternative approach that treats entire document as single text.
        
        Args:
            pages: List of dicts with keys: page, text, char_count, method (optional)
            pdf_name: Name of the source PDF file
            
        Returns:
            List of chunk dicts with metadata
        """
        # Combine all pages into single document
        full_text = "\n\n".join(
            page.get("text", "") for page in pages if page.get("text", "").strip()
        )
        
        if not full_text:
            print("[WARN] No text found in document")
            return []
        
        # Split into chunks
        chunks = self.text_splitter.split_text(full_text)
        
        all_chunks = []
        for chunk_id, chunk_text in enumerate(chunks):
            # Filter small chunks
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            chunk_dict = {
                "text": chunk_text,
                "metadata": {
                    "pdf_name": pdf_name,
                    "page": -1,  # Page info lost in combined approach
                    "chunk_id": chunk_id,
                    "method": "combined",
                    "char_count": len(chunk_text),
                    "source_char_count": len(full_text)
                }
            }
            
            all_chunks.append(chunk_dict)
        
        return all_chunks
    
    def print_chunk_summary(self, chunks: List[Dict], show_first_n: int = 3):
        """
        Print summary statistics and preview of chunks.
        
        Args:
            chunks: List of chunk dictionaries
            show_first_n: Number of chunks to display in full
        """
        if not chunks:
            print("[WARN] No chunks to display")
            return
        
        # Print summary
        print("\n" + "=" * 70)
        print("CHUNKING SUMMARY")
        print("=" * 70)
        print(f"Total chunks: {len(chunks)}")
        
        # Calculate statistics
        total_chars = sum(c["metadata"]["char_count"] for c in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        min_size = min(c["metadata"]["char_count"] for c in chunks)
        max_size = max(c["metadata"]["char_count"] for c in chunks)
        
        print(f"Total characters: {total_chars:,}")
        print(f"Average chunk size: {avg_chunk_size:.0f} chars")
        print(f"Min chunk size: {min_size} chars")
        print(f"Max chunk size: {max_size} chars")
        
        # Group by page
        pages_used = set(c["metadata"]["page"] for c in chunks)
        print(f"Pages with chunks: {len(pages_used)}")
        
        # Group by extraction method
        methods = {}
        for chunk in chunks:
            method = chunk["metadata"]["method"]
            methods[method] = methods.get(method, 0) + 1
        
        print("\nChunks by extraction method:")
        for method, count in methods.items():
            print(f"  {method}: {count}")
        
        # Display first N chunks
        print("\n" + "=" * 70)
        print(f"FIRST {show_first_n} CHUNKS (FULL TEXT)")
        print("=" * 70)
        
        for i, chunk in enumerate(chunks[:show_first_n]):
            metadata = chunk["metadata"]
            print(f"\n{'─' * 70}")
            print(f"Chunk #{metadata['chunk_id']} | Page: {metadata['page']} | "
                  f"Method: {metadata['method']} | Chars: {metadata['char_count']}")
            print(f"{'─' * 70}")
            print(chunk["text"])
        
        print("\n" + "=" * 70)


def main():
    """Demo usage of TextChunker."""
    # Example with Malayalam text
    sample_pages = [
        {
            "page": 1,
            "text": """മലയാളം ഭാഷ വളരെ സുന്ദരമാണ്. ഇത് കേരളത്തിന്റെ ഔദ്യോഗിക ഭാഷയാണ്.
            
മലയാളം ഭാഷയ്ക്ക് സമ്പന്നമായ സാഹിത്യ പാരമ്പര്യമുണ്ട്. നമ്മുടെ സംസ്കാരത്തിന്റെ പ്രതിഫലനമാണ് ഈ ഭാഷ.

കേരളം ദൈവത്തിന്റെ സ്വന്തം നാട് എന്നറിയപ്പെടുന്നു. പ്രകൃതി സൗന്ദര്യത്താൽ സമ്പന്നമായ ഈ നാട് ലോകമെമ്പാടും പ്രസിദ്ധമാണ്.

മലയാളം സാഹിത്യത്തിൽ പല മഹാകവികളുണ്ട്. ഉള്ളൂർ, വള്ളത്തോൾ, കുമാരനാശാൻ തുടങ്ങിയവർ ഇതിൽ പെടും.""",
            "char_count": 450,
            "method": "pypdf"
        },
        {
            "page": 2,
            "text": """Malayalam is one of the 22 scheduled languages of India. It is spoken by over 38 million people worldwide.

The language has a rich history dating back to the 9th century. Malayalam script is derived from the ancient Grantha script.

Modern Malayalam literature has produced many acclaimed authors. Writers like Vaikom Muhammad Basheer, M.T. Vasudevan Nair, and O.V. Vijayan are internationally recognized.

Kerala's literacy rate is among the highest in India. Education and literature have always been valued in Kerala society.""",
            "char_count": 520,
            "method": "ocr"
        },
        {
            "page": 3,
            "text": "Short page.",
            "char_count": 11,
            "method": "pypdf"
        }
    ]
    
    # Create chunker
    chunker = TextChunker(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=200
    )
    
    # Chunk the pages
    chunks = chunker.chunk_pages(sample_pages, pdf_name="sample_malayalam.pdf")
    
    # Print summary and preview
    chunker.print_chunk_summary(chunks, show_first_n=3)
    
    # Alternative: chunk entire document as one
    print("\n\n" + "=" * 70)
    print("ALTERNATIVE: DOCUMENT-LEVEL CHUNKING")
    print("=" * 70)
    
    doc_chunks = chunker.chunk_document(sample_pages, pdf_name="sample_malayalam.pdf")
    print(f"\nTotal chunks (document-level): {len(doc_chunks)}")
    
    return chunks


if __name__ == "__main__":
    main()
