"""
Malayalam-Safe Text Cleaning
Applies minimal cleaning while preserving Malayalam content.
No stopword removal, stemming, or sentence tokenization.
"""

import re
import unicodedata
from typing import List, Dict


class TextCleaner:
    """Clean extracted text while preserving Malayalam characters."""
    
    def __init__(self):
        """Initialize the text cleaner with cleaning patterns."""
        # Page number patterns (English and Malayalam)
        self.page_patterns = [
            r'\bPage\s+\d+\b',           # Page 12
            r'\bപേജ്\s+\d+\b',           # പേജ് 12 (Malayalam "page")
            r'\bതാൾ\s+\d+\b',            # താൾ 12 (Malayalam "page")
            r'^\d+$',                     # Standalone numbers on a line
            r'\[\d+\]',                   # [12]
            r'\(\d+\)',                   # (12)
        ]
        
        # URL patterns
        self.url_pattern = r'https?://\S+|www\.\S+'
        
        # Email patterns
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # OCR artifacts and junk patterns
        self.ocr_junk_patterns = [
            r'[|_]{2,}',                  # Multiple pipes or underscores (combined)
            r'[|\\]{2,}',                 # Multiple pipes or backslashes
            r'_{3,}',                     # Multiple underscores (backup)
            r'-{3,}',                     # Multiple dashes (but keep 1-2)
            r'\.{4,}',                    # Excessive dots (keep ellipsis ...)
            r'\s+[\|\\/]\s+',             # Isolated pipes/slashes
            r'[•●■◆◦]+',                  # Bullet points and symbols
        ]
        
    def normalize_unicode(self, text: str) -> str:
        """
        Apply Unicode NFC normalization.
        Critical for Malayalam text consistency.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize('NFC', text)
    
    def remove_page_numbers(self, text: str) -> str:
        """
        Remove page numbers in English and Malayalam.
        
        Args:
            text: Input text
            
        Returns:
            Text with page numbers removed
        """
        # Enhanced patterns for better coverage
        text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'താൾ\s*\d+', '', text)
        text = re.sub(r'പേജ്\s*\d+', '', text)
        
        # Original patterns for edge cases
        cleaned = text
        for pattern in self.page_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        return cleaned
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs and email addresses.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
        """
        # Remove full URL lines (more aggressive)
        cleaned = re.sub(r'.*http[s]?://\S+.*', '', text)
        
        # Remove any remaining URLs
        cleaned = re.sub(self.url_pattern, '', cleaned)
        
        # Remove email label lines (e.g., "Email: test@example.com")
        cleaned = re.sub(r'.*email\s*:\s*\S+.*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove emails completely (simplified pattern)
        cleaned = re.sub(r'\S+@\S+', '', cleaned)
        
        return cleaned
    
    def remove_ocr_junk(self, text: str) -> str:
        """
        Remove common OCR artifacts and junk characters.
        
        Args:
            text: Input text
            
        Returns:
            Text with OCR junk removed
        """
        cleaned = text
        for pattern in self.ocr_junk_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        return cleaned
    
    def fix_broken_newlines(self, text: str) -> str:
        """
        Fix broken newlines and collapse excessive whitespace.
        Preserves paragraph breaks.
        
        Args:
            text: Input text
            
        Returns:
            Text with fixed newlines
        """
        # Collapse multiple spaces to single space
        cleaned = re.sub(r'[ \t]+', ' ', text)
        
        # Collapse 3+ newlines to 2 (preserve paragraph breaks)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove spaces at start/end of lines
        cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
        cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)
        
        return cleaned
    
    def remove_headers_footers(self, text: str) -> str:
        """
        Remove common header/footer patterns.
        Conservative approach to avoid removing actual content.
        
        Args:
            text: Input text
            
        Returns:
            Text with headers/footers removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip very short lines at document boundaries (likely headers/footers)
            # But keep them if they contain Malayalam characters
            if len(line_stripped) < 3:
                # Check if it contains Malayalam characters
                if any('\u0D00' <= c <= '\u0D7F' for c in line_stripped):
                    cleaned_lines.append(line)
                continue
            
            # Skip common header/footer indicators
            skip = False
            header_footer_indicators = [
                r'^copyright\s+©',
                r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # Dates
                r'^chapter\s+\d+$',
                r'^section\s+\d+$',
            ]
            
            for indicator in header_footer_indicators:
                if re.match(indicator, line_stripped, re.IGNORECASE):
                    skip = True
                    break
            
            if not skip:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization (CRITICAL for Malayalam)
        cleaned = self.normalize_unicode(text)
        
        # Step 2: Remove page numbers
        cleaned = self.remove_page_numbers(cleaned)
        
        # Step 3: Remove URLs and emails
        cleaned = self.remove_urls(cleaned)
        
        # Step 4: Remove OCR junk
        cleaned = self.remove_ocr_junk(cleaned)
        
        # Step 5: Remove headers/footers
        cleaned = self.remove_headers_footers(cleaned)
        
        # Step 6: Fix broken newlines and excessive whitespace
        cleaned = self.fix_broken_newlines(cleaned)
        
        # Final trim
        cleaned = cleaned.strip()
        
        return cleaned
    
    def clean_pages(self, pages: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Clean a list of page dictionaries from OCR extraction.
        
        Args:
            pages: List of dicts with 'page' and 'text' keys
            
        Returns:
            List of dicts with cleaned text
        """
        cleaned_pages = []
        
        for page_data in pages:
            cleaned_page = page_data.copy()
            cleaned_page['text'] = self.clean_text(page_data.get('text', ''))
            
            # Validate cleaned text length
            if len(cleaned_page['text']) < 50:
                print(
                    f"[WARN] Page {cleaned_page['page']} has very little text "
                    f"({len(cleaned_page['text'])} chars). Possible image-only or title page."
                )
            
            cleaned_pages.append(cleaned_page)
        
        return cleaned_pages
    
    def clean_document(self, pages: List[Dict[str, any]]) -> str:
        """
        Clean all pages and combine into single document.
        
        Args:
            pages: List of dicts with 'page' and 'text' keys
            
        Returns:
            Single cleaned document string
        """
        cleaned_pages = self.clean_pages(pages)
        all_text = '\n\n'.join(
            page['text'] for page in cleaned_pages if page['text']
        )
        return all_text


def main():
    """Demo usage of TextCleaner."""
    # Example with Malayalam text
    sample_text = """
    Page 12
    
    മലയാളം ഭാഷ വളരെ സുന്ദരമാണ്.
    This is English text mixed with Malayalam.
    
    താൾ 15
    
    Visit https://example.com for more info.
    Email: test@example.com
    
    ||| OCR junk |||
    ____
    
    കേരളം ദൈവത്തിന്റെ സ്വന്തം നാട്.
    
    
    
    
    More content here...
    """
    
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(sample_text)
    
    print("ORIGINAL TEXT:")
    print("=" * 60)
    print(sample_text)
    print("\n" + "=" * 60)
    print("CLEANED TEXT:")
    print("=" * 60)
    print(cleaned)
    print("\n" + "=" * 60)
    
    # Example with page list
    pages = [
        {"page": 1, "text": "Page 1\n\nമലയാളം content here"},
        {"page": 2, "text": "താൾ 2\n\nMore Malayalam text"},
    ]
    
    cleaned_pages = cleaner.clean_pages(pages)
    
    print("\nCLEANED PAGES (first 2 real pages):")
    for page in cleaned_pages[:2]:
        print(f"\n--- Page {page['page']} ---")
        print(page['text'][:500])


if __name__ == "__main__":
    main()
