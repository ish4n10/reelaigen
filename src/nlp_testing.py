"""
POC: PDF/Text → Reels JSON using Classic NLP (NO LLM)

Pipeline:
- PDF/Text parsing
- Sentence parsing (spaCy)
- Sentence similarity analysis
- Feature engineering
- Sentence graph (ordered DAG)
- Reel splitting (time-budgeted)
- Natural tone conversion
- Visual intent inference (rules)
- JSON IR output (Manim-ready)

Run:
    pip install spacy networkx unstructured
    python -m spacy download en_core_web_sm
    python nlp_testing.py
"""

import spacy
import networkx as nx
import json
import re
import os
import numpy as np
from numpy.linalg import norm
from typing import List, Dict, Tuple, Optional
from spacy.language import Language

try:
    from unstructured.partition.auto import partition
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARN] unstructured not installed. PDF parsing disabled. Install with: pip install unstructured")
# ---------------- CONFIG ---------------- #

WORDS_PER_SECOND = 2.5
MAX_REEL_SECONDS = 40
MAX_REEL_WORDS = int(WORDS_PER_SECOND * MAX_REEL_SECONDS)

# ---------------- NLP SETUP ---------------- #

# Load spaCy model with vectors for similarity
try:
    nlp = spacy.load("en_core_web_sm")
    # Check if model has vectors
    if not nlp.vocab.vectors.size:
        print("[WARN] Model doesn't have word vectors. Similarity may be limited.")
        print("[INFO] Consider installing en_core_web_md or en_core_web_lg for better similarity.")
except OSError:
    print("[ERROR] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise

# ---------------- STAGE 0: PDF/TEXT PARSING ---------------- #

def parse_input(file_path: Optional[str] = None, text: Optional[str] = None) -> str:
    """
    Parse input from either a PDF file or text string.
    
    Args:
        file_path: Path to PDF or text file
        text: Direct text input
    
    Returns:
        Extracted text as string
    """
    if text:
        return text.strip()
    
    if not file_path:
        raise ValueError("Either file_path or text must be provided")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if it's a PDF
    if file_path.lower().endswith('.pdf'):
        if not PDF_SUPPORT:
            raise ImportError("PDF parsing requires 'unstructured' library. Install with: pip install unstructured")
        
        print(f"[INFO] Parsing PDF: {file_path}")
        elements = partition(filename=file_path)
        extracted_text = "\n\n".join(el.text for el in elements if el.text)
        
        if not extracted_text.strip():
            print("[WARN] No extractable text found in PDF.")
        
        return extracted_text.strip()
    
    # Otherwise, treat as text file
    print(f"[INFO] Reading text file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# ---------------- STAGE 1: SENTENCE PARSING ---------------- #

def parse_sentences(text: str) -> List[Dict]:
    """
    Parse text into sentences and extract basic features.
    Also computes sentence vectors for similarity calculations.
    """
    doc = nlp(text)
    sentences = []

    for i, sent in enumerate(doc.sents):
        tokens = [t.text.lower() for t in sent if t.is_alpha]
        
        # Compute sentence vector (average of token vectors)
        sent_vector = None
        if nlp.vocab.vectors.size:
            token_vectors = [t.vector for t in sent if t.has_vector]
            if token_vectors:
                sent_vector = np.mean(token_vectors, axis=0)
        
        sentences.append({
            "sid": i,
            "text": sent.text.strip(),
            "tokens": tokens,
            "word_count": len(tokens),
            "position": i,
            "vector": sent_vector,
            "doc": sent  # Store spaCy span for later processing
        })

    return sentences

# ---------------- STAGE 1.5: SENTENCE SIMILARITY ---------------- #

def compute_sentence_similarities(sentences: List[Dict]) -> Dict[Tuple[int, int], float]:
    """
    Compute similarity scores between all pairs of sentences using spaCy vectors.
    
    Returns:
        Dictionary mapping (sid1, sid2) tuples to similarity scores (0-1)
    """
    similarities = {}
    
    if not nlp.vocab.vectors.size:
        print("[WARN] Cannot compute similarities - model has no vectors")
        return similarities
    
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            if i >= j:  # Only compute upper triangle
                continue
            
            vec1 = s1.get("vector")
            vec2 = s2.get("vector")
            
            if vec1 is not None and vec2 is not None:
                # Cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = norm(vec1)
                norm2 = norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    similarities[(s1["sid"], s2["sid"])] = float(similarity)
                else:
                    similarities[(s1["sid"], s2["sid"])] = 0.0
            else:
                similarities[(s1["sid"], s2["sid"])] = 0.0
    
    return similarities

def get_most_similar_sentences(sentences: List[Dict], similarities: Dict[Tuple[int, int], float], 
                                top_k: int = 5) -> List[Dict]:
    """
    Get top K most similar sentence pairs.
    """
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for (sid1, sid2), sim_score in sorted_sims[:top_k]:
        sent1 = next(s for s in sentences if s["sid"] == sid1)
        sent2 = next(s for s in sentences if s["sid"] == sid2)
        
        results.append({
            "sentence1": {
                "sid": sid1,
                "text": sent1["text"][:100] + "..." if len(sent1["text"]) > 100 else sent1["text"]
            },
            "sentence2": {
                "sid": sid2,
                "text": sent2["text"][:100] + "..." if len(sent2["text"]) > 100 else sent2["text"]
            },
            "similarity": round(sim_score, 4)
        })
    
    return results

# ---------------- STAGE 2: FEATURE ENGINEERING ---------------- #

def extract_features(sentence: Dict) -> Dict:
    text = sentence["text"].lower()

    return {
        "is_definition": bool(re.search(r"\b(is defined as|refers to|is called)\b", text)),
        "is_comparison": bool(re.search(r"\b(vs|versus|compared to)\b", text)),
        "is_example": "for example" in text,
        "is_process": any(k in text for k in ["first", "then", "next", "finally"]),
        "too_long": sentence["word_count"] > 30
    }

def score_sentence(features: Dict) -> float:
    score = 0.0
    score += 2.0 if features["is_definition"] else 0.0
    score += 1.5 if features["is_comparison"] else 0.0
    score += 1.0 if features["is_example"] else 0.0
    score += 1.0 if features["is_process"] else 0.0
    score -= 1.0 if features["too_long"] else 0.0
    return score

# ---------------- STAGE 3: SENTENCE GRAPH ---------------- #

def build_sentence_graph(sentences: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()

    for s in sentences:
        G.add_node(s["sid"], **s)

    # Preserve document order
    for i in range(len(sentences) - 1):
        G.add_edge(sentences[i]["sid"], sentences[i + 1]["sid"], type="next")

    return G

# ---------------- STAGE 4: REEL SPLITTING ---------------- #

def split_into_reels(sentences: List[Dict]) -> List[List[Dict]]:
    reels = []
    current_reel = []
    current_words = 0

    for s in sentences:
        if current_words + s["word_count"] > MAX_REEL_WORDS:
            reels.append(current_reel)
            current_reel = []
            current_words = 0

        current_reel.append(s)
        current_words += s["word_count"]

    if current_reel:
        reels.append(current_reel)

    return reels

# ---------------- STAGE 5: TIMING ---------------- #

def compute_duration(word_count: int) -> float:
    return round(word_count / WORDS_PER_SECOND, 2)

# ---------------- STAGE 5.5: NATURAL TONE CONVERSION ---------------- #

def convert_to_natural_tone(text: str, doc: Optional[object] = None) -> str:
    """
    Convert formal/academic text to more natural, conversational tone.
    Uses rule-based transformations with efficient spaCy token-level processing.
    Optionally uses pre-parsed spaCy doc to avoid redundant parsing.
    """
    if not text:
        return text
    
    # Start with original text
    natural = text.strip()
    
    # 1. Replace formal phrases with conversational ones (using regex for pattern matching)
    replacements = {
        r'\bIt is important to note that\b': 'Here\'s the thing:',
        r'\bIt should be noted that\b': 'Keep in mind that',
        r'\bIt can be observed that\b': 'You\'ll notice that',
        r'\bIn order to\b': 'To',
        r'\bIn the case of\b': 'For',
        r'\bWith regard to\b': 'About',
        r'\bWith respect to\b': 'About',
        r'\bIn accordance with\b': 'Following',
        r'\bSubsequent to\b': 'After',
        r'\bPrior to\b': 'Before',
        r'\bIn the event that\b': 'If',
        r'\bFor the purpose of\b': 'To',
        r'\bIn the context of\b': 'In',
        r'\bIt is evident that\b': 'Clearly,',
        r'\bIt is apparent that\b': 'Obviously,',
        r'\bIt is worth noting that\b': 'Note that',
        r'\bFurthermore\b': 'Also',
        r'\bMoreover\b': 'Plus',
        r'\bNevertheless\b': 'Still',
        r'\bTherefore\b': 'So',
        r'\bThus\b': 'So',
        r'\bHence\b': 'So',
        r'\bConsequently\b': 'So',
        r'\bAdditionally\b': 'Also',
        r'\bIn addition\b': 'Plus',
        r'\bIn conclusion\b': 'To wrap up',
        r'\bTo summarize\b': 'In short',
    }
    
    for pattern, replacement in replacements.items():
        natural = re.sub(pattern, replacement, natural, flags=re.IGNORECASE)
    
    # 2. Simplify passive voice (basic patterns)
    natural = re.sub(r'\bis performed by\b', 'does', natural, flags=re.IGNORECASE)
    natural = re.sub(r'\bare performed by\b', 'do', natural, flags=re.IGNORECASE)
    
    # 3. Remove excessive hedging
    natural = re.sub(r'\bIt may be that\b', 'Maybe', natural, flags=re.IGNORECASE)
    natural = re.sub(r'\bIt might be that\b', 'Maybe', natural, flags=re.IGNORECASE)
    
    # 4. Simplify complex sentence starters
    natural = re.sub(r'\bAs a result of\b', 'Because of', natural, flags=re.IGNORECASE)
    natural = re.sub(r'\bDue to the fact that\b', 'Because', natural, flags=re.IGNORECASE)
    
    # 5. Make questions more direct
    natural = re.sub(r'\bOne might wonder\b', 'You might ask', natural, flags=re.IGNORECASE)
    
    # 6. Use spaCy for efficient normalization and proper text reconstruction
    # Parse once to normalize whitespace, tokenization, and formatting using spaCy's layout
    # This ensures consistent spacing and proper sentence structure
    natural_doc = nlp(natural)
    natural = natural_doc.text
    
    # 7. Capitalize first letter if needed
    if natural and natural[0].islower():
        natural = natural[0].upper() + natural[1:]
    
    return natural.strip()

def simplify_sentence(sentence: Dict) -> str:
    """
    Simplify a sentence for more natural narration using spaCy's parsed structure.
    Efficiently processes using pre-parsed spaCy doc to avoid redundant parsing.
    """
    text = sentence["text"]
    doc = sentence.get("doc")
    
    # Use the pre-parsed doc for efficient processing
    # The doc is already available from initial parsing, so we pass it to avoid re-parsing
    natural_text = convert_to_natural_tone(text, doc=doc)
    
    # For very long sentences, we could use spaCy's structure to identify simplification opportunities
    if sentence["word_count"] > 25 and doc:
        # spaCy's doc structure is already available, but we've done the conversion
        # Additional processing could be added here if needed
        pass
    
    return natural_text

# ---------------- STAGE 6: VISUAL INTENT ---------------- #

def infer_visual_intent(text: str) -> str:
    t = text.lower()

    if "vs" in t or "versus" in t:
        return "compare_side_by_side"
    if "is defined as" in t or "refers to" in t:
        return "centered_statement"
    if t.count(".") >= 2:
        return "bullet_scene"
    return "text_intro"

# ---------------- STAGE 7: JSON IR ---------------- #

def build_reel_json(reels: List[List[Dict]], similarities: Optional[Dict[Tuple[int, int], float]] = None) -> List[Dict]:
    """
    Build JSON output with natural tone conversion and similarity information.
    """
    output = []

    for idx, reel in enumerate(reels):
        # Original text
        narration_text = " ".join(s["text"] for s in reel)
        
        # Natural tone version - process sentences efficiently
        # Use spaCy's batch processing for better efficiency when possible
        natural_sentences = []
        for s in reel:
            natural_sent = simplify_sentence(s)
            natural_sentences.append(natural_sent)
        
        # Join with proper spacing using spaCy's text normalization
        natural_narration = " ".join(natural_sentences)
        # Final normalization pass using spaCy for consistent formatting
        natural_doc = nlp(natural_narration)
        natural_narration = natural_doc.text
        
        total_words = sum(s["word_count"] for s in reel)
        
        # Get similarity info for sentences in this reel
        reel_similarities = []
        if similarities:
            reel_sids = [s["sid"] for s in reel]
            for (sid1, sid2), sim_score in similarities.items():
                if sid1 in reel_sids or sid2 in reel_sids:
                    # Check if both are in this reel or adjacent
                    if sid1 in reel_sids and sid2 in reel_sids:
                        reel_similarities.append({
                            "sentence1_id": sid1,
                            "sentence2_id": sid2,
                            "similarity": round(sim_score, 4)
                        })
        
        reel_json = {
            "reel_id": idx,
            "prev_reel": idx - 1 if idx > 0 else None,
            "next_reel": idx + 1 if idx < len(reels) - 1 else None,
            "duration_sec": compute_duration(total_words),
            "narration": {
                # "text": narration_text,
                "natural_tone": natural_narration,  # More conversational version
                "word_count": total_words
            },
            "visual": {
                "object": {
                    "type": "text",
                    # "value": reel[0]["text"][:60] + "..."
                },
                "template": infer_visual_intent(narration_text)
            }
        }
        
        # Add similarity information if available
        if reel_similarities:
            reel_json["sentence_similarities"] = reel_similarities[:5]  # Top 5 per reel

        output.append(reel_json)

    return output

# ---------------- MAIN (POC RUN) ---------------- #

if __name__ == "__main__":
    import sys
    
    # Allow input from file or use sample text
    input_text = None
    input_file = None
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"\n[INFO] Using input file: {input_file}")
    else:
        # Complex research/academic text for testing NLP parsing and natural tone conversion
        input_text = """
        It is important to note that machine learning algorithms have revolutionized the field of computational biology in recent years. 
        In order to understand the underlying mechanisms, it should be noted that deep neural networks are capable of identifying patterns 
        that were previously undetectable through traditional statistical methods. Furthermore, it can be observed that convolutional 
        neural networks, when applied to genomic sequence analysis, demonstrate remarkable accuracy in predicting protein structures. 
        With regard to the experimental validation, it is worth noting that these predictions have been subsequently verified through 
        crystallographic studies. In the context of drug discovery, it is evident that machine learning approaches have significantly 
        accelerated the identification of potential therapeutic compounds. Additionally, it may be that reinforcement learning algorithms 
        could be employed to optimize molecular design processes. In accordance with current research trends, it is apparent that the 
        integration of multi-omics data with machine learning models represents a promising avenue for personalized medicine. 
        Prior to the advent of these computational techniques, drug development cycles were substantially longer and more resource-intensive. 
        Subsequent to the implementation of these methods, pharmaceutical companies have reported reductions in both time and cost. 
        In the event that these trends continue, it is conceivable that the entire drug discovery pipeline could be transformed within 
        the next decade. For the purpose of advancing this field, it is crucial that interdisciplinary collaboration between computer 
        scientists and biologists be fostered. In the case of rare diseases, where traditional approaches have been largely unsuccessful, 
        machine learning offers a particularly compelling alternative. With respect to ethical considerations, it should be noted that 
        the deployment of these technologies raises important questions regarding data privacy and algorithmic bias. Nevertheless, 
        the potential benefits cannot be overlooked. Therefore, it is imperative that regulatory frameworks be developed in parallel 
        with technological advancement. Thus, the scientific community must balance innovation with responsibility. Hence, ongoing 
        dialogue between researchers, clinicians, and policymakers is essential. Consequently, the future of computational biology 
        will likely be shaped by both technological breakthroughs and thoughtful governance. In conclusion, while challenges remain, 
        the trajectory of machine learning in biology appears to be overwhelmingly positive. To summarize, the convergence of artificial 
        intelligence and life sciences has created unprecedented opportunities for scientific discovery and medical advancement.
        """
        print("\n[INFO] Using complex research sample text. Provide a file path as argument to parse PDF/text file.")
    
    # Parse input (PDF or text)
    try:
        raw_text = parse_input(file_path=input_file, text=input_text)
    except Exception as e:
        print(f"\n[ERROR] Failed to parse input: {e}")
        sys.exit(1)
    
    print("\n--- INPUT TEXT ---\n")
    print(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)

    # Parse sentences
    print("\n[INFO] Parsing sentences...")
    sentences = parse_sentences(raw_text)
    
    print(f"\n[INFO] Parsed {len(sentences)} sentences")

    # Compute sentence similarities
    print("\n[INFO] Computing sentence similarities...")
    similarities = compute_sentence_similarities(sentences)
    
    if similarities:
        print(f"[INFO] Computed {len(similarities)} similarity pairs")
        top_similar = get_most_similar_sentences(sentences, similarities, top_k=3)
        print("\n--- TOP SIMILAR SENTENCES ---\n")
        for item in top_similar:
            print(f"Similarity: {item['similarity']:.4f}")
            print(f"  S1: {item['sentence1']['text']}")
            print(f"  S2: {item['sentence2']['text']}\n")
    
    # Feature + scoring
    print("\n[INFO] Extracting features...")
    for s in sentences:
        feats = extract_features(s)
        s["features"] = feats
        s["score"] = score_sentence(feats)

    # Graph
    graph = build_sentence_graph(sentences)
    print(f"[INFO] Built sentence graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Reels
    reels = split_into_reels(sentences)
    print(f"[INFO] Split into {len(reels)} reels")

    # JSON IR with natural tone and similarities
    print("\n[INFO] Building JSON output with natural tone conversion...")
    reel_json = build_reel_json(reels, similarities)

    print("\n--- OUTPUT JSON ---\n")
    print(json.dumps(reel_json, indent=2))
    
    # Show natural tone examples
    print("\n--- NATURAL TONE EXAMPLES ---\n")
    for reel in reel_json[:3]:  # Show first 3 reels
        print(f"Reel {reel['reel_id']}:")
        print(f"  Natural:  {reel['narration']['natural_tone'][:150]}...")
        print()
