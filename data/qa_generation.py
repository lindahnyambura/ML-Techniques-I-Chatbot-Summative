import json
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM

class QAGenerator:
    def __init__(self):
        # Initialize with small model that fits in memory
        self.model_name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.qa_model = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # Use CPU (-1) instead of string 'cpu'
        )
        self.output_dir = Path("data/qa_pairs/automated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_qa_from_text(self, text_chunk, source_label):
        """Generate one QA pair per call with more reliable prompt"""
        prompt = f"""
        Create one question and answer pair based on this text.
        ALWAYS use exactly this format:
        Question: [question here]
        Answer: [answer here]
        
        Text: {text_chunk[:500]}  # Smaller chunk for flan-t5-small
        """
        
        try:
            result = self.qa_model(
                prompt,
                max_length=200,  # Reduced for small model
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )
            return self._parse_qa_response(result[0]["generated_text"], source_label)
        except Exception as e:
            print(f"Error generating QA: {str(e)[:100]}...")  # Truncate long errors
            return []

    def _parse_qa_response(self, response_text, source_label):
        """More robust parsing that handles malformed outputs"""
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        qa_pairs = []
        
        # Look for Question/Answer pairs in any order
        for i in range(len(lines)-1):
            if lines[i].startswith("Question:") and lines[i+1].startswith("Answer:"):
                q = lines[i][len("Question:"):].strip()
                a = lines[i+1][len("Answer:"):].strip()
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "source": source_label,
                    "type": "automated"
                })
        return qa_pairs

    def process_all_segments(self):
        """Process files with error handling and progress tracking"""
        for book_dir in Path("data/cleaned_text").glob("*_cleaned"):
            if not book_dir.is_dir():
                continue
                
            qa_data = []
            print(f"\nProcessing {book_dir.name}...")
            seg_files = list(book_dir.glob("*.txt"))
            
            for seg_file in tqdm(seg_files, desc="Generating QAs"):
                try:
                    with open(seg_file, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text.strip():  # Skip empty files
                            qa_data.extend(self.generate_qa_from_text(text, book_dir.name))
                except Exception as e:
                    print(f"Error processing {seg_file.name}: {str(e)[:100]}...")
                    continue
            
            # Save progress after each book
            if qa_data:  # Only save if we got data
                output_file = self.output_dir / f"{book_dir.name}_qa.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(qa_data, f, indent=2)
                print(f"Saved {len(qa_data)} QA pairs to {output_file}")

if __name__ == "__main__":
    generator = QAGenerator()
    generator.process_all_segments()