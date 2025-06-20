import re
import json
from datetime import datetime
from pathlib import Path

class TimelineExtractor:
    def __init__(self):
        self.output_dir = Path("data/knowledge_base/timelines")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events = []

    def extract_events(self, text, doc_label="source"):
        # Regex to find dates and events
        date_patterns = [
            r"(\d{1,2}(?:st|nd|rd|th)? \w+ \d{4})",  # "21 October 1956"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}",  # "October 1956"
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                date_str = match.group()
                context = text[match.start()-50 : match.end()+50]  # Nearby text
                
                self.events.append({
                    "date": date_str,
                    "event": context.strip(),
                    "source": doc_label
                })

    def save_timeline(self):
        output_file = self.output_dir / "kimathi_timeline.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, indent=2)

if __name__ == "__main__":
    extractor = TimelineExtractor()
    
    cleaned_files = [
        ("dedan_kimathi_on_trial", "data/cleaned_text/dedan_kimathi_on_trial_cleaned.txt"),
        ("the_trial_of_dedan_kimathi", "data/cleaned_text/the_trial_of_dedan_kimathi_cleaned.txt")
    ]
    
    for label, filepath in cleaned_files:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                extractor.extract_events(text, label)
    
    extractor.save_timeline()
    print("Timeline extraction complete. Check /knowledge_base/timelines/")