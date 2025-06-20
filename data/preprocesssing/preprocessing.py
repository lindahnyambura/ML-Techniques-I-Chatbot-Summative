import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
import re
from text_cleaner import KimathiTextCleaner
from pathlib import Path

class PDFProcessor:
    def __init__(self, input_dir="data/raw_text", output_dir="data/extracted_text"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def extract_text_from_text_based_pdf(self, pdf_path):
        """Extract text from normal PDFs"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in tqdm(pdf.pages, desc=f"Processing {os.path.basename(pdf_path)}"):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()
    
    def extract_text_from_scanned_pdf(self, pdf_path):
        """Extract text from scanned PDFs using OCR"""
        text = ""
        images = convert_from_path(pdf_path, dpi=300)

        custom_config = r'--oem 3 --psm 6 -l eng+swa'

        
        for i, image in enumerate(tqdm(images, desc=f"OCR Processing {os.path.basename(pdf_path)}")):
            # preprocess image
            image = image.convert('L') # grayscale
            image = image.point(lambda x: 0 if x < 140 else 255)    # thresholding
            page_text = pytesseract.image_to_string(image, config=custom_config)
            text += page_text + "\n\n"
        
        return text.strip()
    
    def clean_extracted_text(self, text):
        """Basic cleaning of extracted text"""
        # Fix common OCR artifacts
        replacements = {
            r'•·([a-z])': r'\1', # fix "•·oices" -> "voices"
            r'�': 'u',   # replacement for umlaut
            r'([a-z])_([a-z])': r'\1\2',    # fix joined words
            r'([A-Z])_([a-z])': r'\1\2',    # fix joined words with capital
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Preserve intentional line breaks in poetry/dialogue
        text = re.sub(r'(\S)\n(\S)', r'\1 \2', text) # jojn broken lines
        text = re.sub(r'(\S)\n\n(\S)', r'\1\n\n\2', text)  #keep intentional breaks
        # whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def process_all_pdfs(self):
        """Process all PDFs in the input directory"""
        results = {}
        
        for pdf_file in os.listdir(self.input_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(self.input_dir, pdf_file)
                output_path = os.path.join(self.output_dir, f"{os.path.splitext(pdf_file)[0]}.txt")
                
                print(f"\nProcessing: {pdf_file}")
                
                try:
                    # Try text extraction first
                    text = self.extract_text_from_text_based_pdf(pdf_path)
                    
                    # If little text was extracted, try OCR
                    if len(text.split()) < 100:  # heuristic for scanned PDF
                        print("Detected scanned PDF, switching to OCR...")
                        text = self.extract_text_from_scanned_pdf(pdf_path)
                    
                    # Clean and save
                    cleaned_text = self.clean_extracted_text(text)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    
                    results[pdf_file] = {
                        'status': 'success',
                        'output_path': output_path,
                        'word_count': len(cleaned_text.split())
                    }
                except Exception as e:
                    results[pdf_file] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        return results

if __name__ == "__main__":
    processor = PDFProcessor()
    results = processor.process_all_pdfs()
    
    print("\nProcessing Summary:")
    for file, result in results.items():
        status = result['status']
        print(f"{file}: {status.upper()}")
        if status == 'success':
            print(f"  - Saved to: {result['output_path']}")
            print(f"  - Word count: {result['word_count']}")
        else:
            print(f"  - Error: {result['error']}")



class TextPreprocessor:
    def __init__(self):
        self.cleaner = KimathiTextCleaner()
        self.output_dir = Path("data/cleaned_text")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def segment_text(self, text, max_chars=2000):
        """Split text into logical chunks"""
        segments = []
        current_segment = ""
        
        # First split by major sections
        for section in re.split(r'\n{2,}', text):
            if len(current_segment) + len(section) <= max_chars:
                current_segment += "\n\n" + section
            else:
                # Split by sentences if section is too long
                sentences = re.split(r'(?<=[.!?])\s+', section)
                for sentence in sentences:
                    if len(current_segment) + len(sentence) <= max_chars:
                        current_segment += " " + sentence
                    else:
                        if current_segment.strip():
                            segments.append(current_segment.strip())
                        current_segment = sentence
        
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return segments

    def process_book(self, input_path, book_name):
        """Process a single book"""
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean text
        cleaned_text = self.cleaner.clean_text(text)
        
        # Save full cleaned text
        combined_path = self.output_dir / f"{book_name}_cleaned.txt"
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Segment and save sections
        segments = self.segment_text(cleaned_text)
        book_dir = self.output_dir / f"{book_name}_cleaned"
        book_dir.mkdir(exist_ok=True)
        
        for i, segment in enumerate(segments, 1):
            segment_path = book_dir / f"section_{i:03d}.txt"
            with open(segment_path, 'w', encoding='utf-8') as f:
                f.write(segment)
        
        return len(segments)

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Process both books
    books = {
        "dedan_kimathi_on_trial": "data/extracted_text/dedan_kimathi_on_trial.txt",
        "the_trial_of_dedan_kimathi": "data/extracted_text/the_trial_of_dedan_kimathi.txt"
    }
    
    print("Starting text cleaning and segmentation...")
    for book_name, input_path in books.items():
        if os.path.exists(input_path):
            num_segments = preprocessor.process_book(input_path, book_name)
            print(f"Processed {book_name}: {num_segments} segments created")
        else:
            print(f"File not found: {input_path}")