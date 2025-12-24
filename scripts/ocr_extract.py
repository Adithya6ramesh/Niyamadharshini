"""
PDF Text Extraction with OCR Fallback
Handles Malayalam and other languages safely.
Uses PyPDF for text extraction, falls back to PaddleOCR when needed.
"""

import os
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import pytesseract


class PDFTextExtractor:
    """Extract text from PDFs with intelligent OCR fallback."""
    
    def __init__(self, min_text_threshold: int = 50):
        """
        Initialize the PDF text extractor.
        
        Args:
            min_text_threshold: Minimum text length to consider extraction successful
        """
        self.min_text_threshold = min_text_threshold
        self.ocr = None  # Lazy initialization
        
    def _init_ocr(self):
        """Initialize PaddleOCR lazily (only when needed)."""
        if self.ocr is None:
          self.ocr = PaddleOCR(
          use_textline_orientation=True,
          lang="en"
        )
    
    def _contains_malayalam(self, text: str) -> bool:
        """Check if text contains Malayalam characters."""
        return any('\u0D00' <= c <= '\u0D7F' for c in text)
    
    def _ocr_with_tesseract(self, img: Image.Image) -> str:
        """OCR using Tesseract with Malayalam support."""
        try:
            text = pytesseract.image_to_string(
                img,
                lang="mal+eng",
                config="--psm 6"
            )
            return text.strip()
        except Exception as e:
            print(f"[ERROR] Tesseract OCR failed: {e}")
            return ""
    
    def _extract_text_pypdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text using PyMuPDF (fitz).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dicts with page number and extracted text
        """
        results = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                results.append({
                    "page": page_num + 1,
                    "text": text.strip()
                })
            doc.close()
        except Exception as e:
            print(f"Error extracting text with PyMuPDF: {e}")
        return results
    
    def _extract_text_ocr_page(self, doc, page_num: int) -> str:
        """
        Extract text from a single PDF page using dual OCR (PaddleOCR + Tesseract).
        
        Args:
            doc: Opened fitz document object
            page_num: Page number (0-indexed)
            
        Returns:
            Extracted text
        """
        page = doc[page_num]

        mat = fitz.Matrix(2.5, 2.5)
        pix = page.get_pixmap(matrix=mat)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 1️⃣ Try PaddleOCR first (good for layouts)
        self._init_ocr()
        try:
            paddle_result = self.ocr.ocr(np.array(img))
            paddle_text = []

            if isinstance(paddle_result, list):
                for block in paddle_result:
                    if not block:
                        continue
                    for line in block:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            txt = line[1][0]
                            if txt.strip():
                                paddle_text.append(txt)

            paddle_text = "\n".join(paddle_text)

            if len(paddle_text) >= self.min_text_threshold:
                return paddle_text

        except Exception:
            pass  # silently fallback

        # 2️⃣ Fallback to Tesseract (Malayalam-safe)
        tess_text = self._ocr_with_tesseract(img)
        return tess_text
    
    def _needs_ocr(self, text: str) -> bool:
        """
        Determine if OCR is needed based on extracted text quality.
        
        Args:
            text: Extracted text
            
        Returns:
            True if OCR is needed, False otherwise
        """
        if not text.strip():
            return True
        if len(text.strip()) < self.min_text_threshold:
            return True
        return False
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF with intelligent fallback to OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dicts: [{"page": int, "text": str}, ...]
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Processing: {pdf_path}")
        
        # First attempt: PyMuPDF text extraction
        results = self._extract_text_pypdf(pdf_path)
        
        # Open document once for OCR processing
        doc = fitz.open(pdf_path)
        
        # Check each page and apply OCR if needed
        for i, page_data in enumerate(results):
            if self._needs_ocr(page_data["text"]):
                print(f"  Page {page_data['page']}: Text extraction insufficient, using OCR...")
                ocr_text = self._extract_text_ocr_page(doc, i)
                
                if ocr_text and len(ocr_text.strip()) >= self.min_text_threshold:
                    # OCR worked → use it
                    page_data["text"] = ocr_text
                    page_data["method"] = "ocr"
                else:
                    # OCR failed → KEEP PyPDF text
                    print(f"  Page {page_data['page']}: OCR failed, keeping PyMuPDF text")
                    page_data["method"] = "pypdf_fallback"
            else:
                print(f"  Page {page_data['page']}: Text extracted successfully")
                page_data["method"] = "pypdf"
            
            # Add character count metadata
            page_data["char_count"] = len(page_data["text"])
        
        doc.close()
        
        return results
    
    def extract_from_directory(self, data_dir: str) -> Dict[str, List[Dict[str, any]]]:
        """
        Extract text from all PDFs in a directory.
        
        Args:
            data_dir: Directory containing PDF files
            
        Returns:
            Dict mapping filename to extracted page data
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        pdf_files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.PDF"))
        
        if not pdf_files:
            print(f"No PDF files found in {data_dir}")
            return {}
        
        all_results = {}
        
        for pdf_file in pdf_files:
            try:
                results = self.extract_from_pdf(str(pdf_file))
                all_results[pdf_file.name] = results
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
        
        return all_results


def main():
    """Main function for testing the extractor."""
    import sys
    
    # Default to data directory
    data_dir = "data"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    # Create extractor
    extractor = PDFTextExtractor(min_text_threshold=50)
    
    # Process all PDFs in directory
    results = extractor.extract_from_directory(data_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    for filename, pages in results.items():
        total_chars = sum(len(page["text"]) for page in pages)
        ocr_pages = sum(1 for page in pages if page.get("method") == "ocr")
        pypdf_pages = sum(1 for page in pages if page.get("method") == "pypdf")
        
        print(f"\n{filename}:")
        print(f"  Total pages: {len(pages)}")
        print(f"  PyPDF pages: {pypdf_pages}")
        print(f"  OCR pages: {ocr_pages}")
        print(f"  Total characters: {total_chars:,}")
        
        # Show sample from first page
        if pages and pages[0]["text"]:
            sample = pages[0]["text"][:200].replace("\n", " ")
            print(f"  Sample: {sample}...")
    
    return results


if __name__ == "__main__":
    main()
