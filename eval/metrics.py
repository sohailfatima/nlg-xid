from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import re

def tokenize(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s.lower())

def bleu_score(preds: List[str], refs: List[List[str]], ngram: int = 4) -> float:
    # refs: list of list(s) of reference tokens
    preds_tok = [tokenize(p) for p in preds]
    refs_tok = [[tokenize(r) for r in rlist] for rlist in refs]
    chencherry = SmoothingFunction()
    score = corpus_bleu(refs_tok, preds_tok, smoothing_function=chencherry.method1, weights=tuple([1/ngram]*ngram))
    return float(score)

def shap_fidelity_check(explanations: List[str], top_features_lists: List[List[str]]) -> float:
    # Simple proxy: fraction of explanations that mention at least half of top features by name
    hits = 0
    for text, tops in zip(explanations, top_features_lists):
        count = sum(1 for f in tops if f.lower() in text.lower())
        if count >= max(1, len(tops)//2):
            hits += 1
    return hits / max(1, len(explanations))
