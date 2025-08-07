from typing import List, Dict, Any, Tuple
import json, re
from explain.llm_explainer import OpenRouterClient

JUDGE_PROMPT = """You are an impartial evaluator for SOC explanations.
Given a prediction, SHAP top features (with signs), and a candidate explanation, score:
1) Fidelity (0-5): aligns with features/directions; penalize contradictions/hallucinations.
2) Clarity (0-5): concise, unambiguous, SOC-ready.
3) Completeness (0-5): justifies label/anomaly; covers key drivers; notes caveats.
Respond ONLY as JSON: {"fidelity": float, "clarity": float, "completeness": float, "rationale": "..."}
"""

def _build_case_prompt(label: str, top_feats: List[Tuple[str, float]], explanation: str, instance_snippet: str = "") -> str:
    tf_str = ", ".join([f"{k} ({v:+.3f})" for k, v in top_feats]) if top_feats else "N/A"
    return "\n\n".join([JUDGE_PROMPT.strip(), f"Prediction: {label}", f"Top features: {tf_str}", f"Instance: {instance_snippet}", f"Explanation: {explanation}"])

def _heuristic_scores(expl: str, label: str, top_feats: List[Tuple[str, float]]) -> Dict[str, float]:
    text = (expl or ""); lower = text.lower()
    feats = [k for k,_ in top_feats]; hits = sum(1 for k in feats if k.lower() in lower); denom = max(1,len(feats))
    fidelity = 5.0 * (hits/denom)
    sents = re.split(r"[.!?]+", text); n = max(1, sum(1 for s in sents if s.strip()))
    avg_len = sum(len(s.split()) for s in sents if s.strip()) / n; has_punct = bool(re.search(r"[.!?]", text))
    clarity = 5.0; 
    if avg_len>28: clarity-=1.0
    if avg_len>40: clarity-=1.0
    if not has_punct: clarity-=1.0
    clarity = max(0.0, min(5.0, clarity))
    evidence_tokens = hits + (1 if re.search(r"\b(because|due to|driven by|key factor|reason)\b", lower) else 0)
    completeness = 2.0 + min(3.0, evidence_tokens*0.75)
    if label and label.lower() in lower: completeness = min(5.0, completeness + 0.5)
    return {"fidelity": round(fidelity,3), "clarity": round(clarity,3), "completeness": round(completeness,3), "rationale": "heuristic-stub"}

class LLMJudge:
    def __init__(self, provider: str='stub', model: str='gpt-4o-mini', weights: Dict[str,float]=None, api_key: str=None):
        self.provider = provider; self.model = model
        self.client = OpenRouterClient(model=model, api_key=api_key)
        self.weights = weights or {"fidelity":0.5,"clarity":0.25,"completeness":0.25}

    def score_cases(self, labels: List[str], explanations: List[str], top_features_lists: List[List[str]], shap_values_per_case: List[Dict[str,float]], instance_snippets: List[str]=None) -> Dict[str,Any]:
        out = {"cases": [], "summary": {}}
        instance_snippets = instance_snippets or [""]*len(explanations)
        totals = {"fidelity":0.0,"clarity":0.0,"completeness":0.0,"overall":0.0}; n = max(1,len(explanations))
        for label, expl, tops, shapd, snip in zip(labels, explanations, top_features_lists, shap_values_per_case, instance_snippets):
            tf_pairs = [(f, float(shapd.get(f,0.0))) for f in tops]
            if self.provider=='stub':
                scores = _heuristic_scores(expl, label, tf_pairs)
            else:
                prompt = _build_case_prompt(label, tf_pairs, expl, snip)
                raw = self.client.generate(prompt)
                try: scores = json.loads(raw)
                except Exception: scores = _heuristic_scores(expl, label, tf_pairs)
            overall = (self.weights['fidelity']*scores['fidelity'] + self.weights['clarity']*scores['clarity'] + self.weights['completeness']*scores['completeness'])
            scores['overall'] = round(overall,3)
            out['cases'].append(scores)
            for k in totals: totals[k]+=scores.get(k,0.0)
        out['summary'] = {k: round(v/n,3) for k,v in totals.items()}
        return out
