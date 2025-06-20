import spacy
import networkx as nx
import json
from pathlib import Path

nlp = spacy.load("en_core_web_lg")

class RelationshipExtractor:
    def __init__(self):
        self.output_dir = Path("data/knowledge_base/relationships")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph = nx.Graph()

    def extract_relationships(self, text, doc_label="source"):
        doc = nlp(text)
        
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "dobj", "pobj"):
                    subject = token.head.text
                    obj = token.text
                    relation = token.dep_
                    
                    self.graph.add_edge(subject, obj, relation=relation, source=doc_label)

    def save_relationships(self):
        output_file = self.output_dir / "kimathi_network.json"
        relationships = [
            {
                "source": u,
                "target": v,
                "relation": data["relation"],
                "context": data["source"]
            }
            for u, v, data in self.graph.edges(data=True)
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(relationships, f, indent=2)

if __name__ == "__main__":
    extractor = RelationshipExtractor()
    
    cleaned_files = [
        ("dedan_kimathi_on_trial", "data/cleaned_text/dedan_kimathi_on_trial_cleaned.txt"),
        ("the_trial_of_dedan_kimathi", "data/cleaned_text/the_trial_of_dedan_kimathi_cleaned.txt")
    ]
    
    for label, filepath in cleaned_files:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                extractor.extract_relationships(text, label)
    
    extractor.save_relationships()
    print("Relationship extraction complete. Check /knowledge_base/relationships/")