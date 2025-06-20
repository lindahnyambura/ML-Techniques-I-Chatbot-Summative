from sklearn.feature_extraction.text import TfidfVectorizer
import json
from pathlib import Path

class ThemeExtractor:
    def __init__(self):
        self.output_dir = Path("data/knowledge_base/themes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.themes = []

    def extract_themes(self, text, doc_label="source"):
        # Use TF-IDF to find important terms
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = tfidf.fit_transform([text])
        feature_names = tfidf.get_feature_names_out()
        
        self.themes.append({
            "source": doc_label,
            "key_terms": feature_names.tolist()
        })

    def save_themes(self):
        output_file = self.output_dir / "mau_mau_themes.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.themes, f, indent=2)

if __name__ == "__main__":
    extractor = ThemeExtractor()
    
    cleaned_files = [
        ("dedan_kimathi_on_trial", "data/cleaned_text/dedan_kimathi_on_trial_cleaned.txt"),
        ("the_trial_of_dedan_kimathi", "data/cleaned_text/the_trial_of_dedan_kimathi_cleaned.txt")
    ]
    
    for label, filepath in cleaned_files:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                extractor.extract_themes(text, label)
    
    extractor.save_themes()
    print("Theme extraction complete. Check /knowledge_base/themes/")