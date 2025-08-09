from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import re
import numpy as np
from collections import Counter

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

def ranked_fidelity_score(explanations: List[str], shap_dicts: List[Dict[str, float]], 
                         top_k: int = 5) -> float:
    """
    Better fidelity metric: Check if explanations mention features in order of SHAP importance.
    Uses Spearman rank correlation between SHAP importance and mention order in text.
    """
    if not explanations or not shap_dicts:
        return 0.0
    
    total_correlation = 0.0
    valid_samples = 0
    
    for explanation, shap_values in zip(explanations, shap_dicts):
        # Get top-k most important features by absolute SHAP value
        sorted_features = sorted(shap_values.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:top_k]
        
        if not sorted_features:
            continue
            
        # Find order of mention in explanation text
        feature_positions = {}
        explanation_lower = explanation.lower()
        
        for feature_name, _ in sorted_features:
            # Look for feature name in explanation
            pos = explanation_lower.find(feature_name.lower())
            if pos != -1:
                feature_positions[feature_name] = pos
        
        # Need at least 2 features mentioned to compute correlation
        if len(feature_positions) < 2:
            continue
            
        # Get SHAP ranks and mention position ranks
        mentioned_features = list(feature_positions.keys())
        shap_ranks = [i for i, (feat, _) in enumerate(sorted_features) 
                     if feat in mentioned_features]
        position_ranks = sorted(range(len(mentioned_features)), 
                              key=lambda i: feature_positions[mentioned_features[i]])
        
        # Compute Spearman correlation
        if len(shap_ranks) > 1:
            correlation = np.corrcoef(shap_ranks, position_ranks)[0, 1]
            if not np.isnan(correlation):
                total_correlation += correlation
                valid_samples += 1
    
    return total_correlation / max(1, valid_samples)

def feature_coverage_fidelity(explanations: List[str], shap_dicts: List[Dict[str, float]],
                             importance_threshold: float = 0.1) -> Dict[str, float]:
    """
    Measures what fraction of important features are covered in explanations.
    Returns precision, recall, and F1 for feature coverage.
    """
    if not explanations or not shap_dicts:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    total_precision = 0.0
    total_recall = 0.0
    valid_samples = 0
    
    for explanation, shap_values in zip(explanations, shap_dicts):
        # Get important features (above threshold)
        important_features = {feat for feat, val in shap_values.items() 
                            if abs(val) >= importance_threshold}
        
        if not important_features:
            continue
            
        # Find mentioned features in explanation
        explanation_lower = explanation.lower()
        mentioned_features = {feat for feat in shap_values.keys() 
                            if feat.lower() in explanation_lower}
        
        if mentioned_features:
            # Precision: fraction of mentioned features that are actually important
            precision = len(mentioned_features & important_features) / len(mentioned_features)
            # Recall: fraction of important features that are mentioned
            recall = len(mentioned_features & important_features) / len(important_features)
            
            total_precision += precision
            total_recall += recall
            valid_samples += 1
    
    if valid_samples == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    avg_precision = total_precision / valid_samples
    avg_recall = total_recall / valid_samples
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    
    return {'precision': avg_precision, 'recall': avg_recall, 'f1': f1}

def importance_consistency_score(explanations: List[str], shap_dicts: List[Dict[str, float]]) -> float:
    """
    Check if explanations correctly identify the direction of feature impact.
    Looks for positive/negative language around feature mentions and compares to SHAP sign.
    """
    if not explanations or not shap_dicts:
        return 0.0
    
    # Keywords indicating positive/negative impact
    positive_words = {'high', 'large', 'increase', 'elevated', 'significant', 'strong', 'contributes', 'indicates'}
    negative_words = {'low', 'small', 'decrease', 'reduced', 'minimal', 'weak', 'prevents', 'normal'}
    
    total_consistency = 0.0
    valid_samples = 0
    
    for explanation, shap_values in zip(explanations, shap_dicts):
        explanation_lower = explanation.lower()
        feature_consistencies = []
        
        for feature_name, shap_val in shap_values.items():
            if abs(shap_val) < 0.05:  # Skip features with minimal impact
                continue
                
            # Find feature mention in text
            feature_pos = explanation_lower.find(feature_name.lower())
            if feature_pos == -1:
                continue
            
            # Look for sentiment words in context (±50 characters around feature)
            context_start = max(0, feature_pos - 50)
            context_end = min(len(explanation_lower), feature_pos + len(feature_name) + 50)
            context = explanation_lower[context_start:context_end]
            
            # Count positive/negative sentiment
            pos_count = sum(1 for word in positive_words if word in context)
            neg_count = sum(1 for word in negative_words if word in context)
            
            if pos_count == 0 and neg_count == 0:
                continue  # No sentiment detected
            
            # Determine explanation sentiment
            explanation_sentiment = 1 if pos_count > neg_count else -1
            shap_sentiment = 1 if shap_val > 0 else -1
            
            # Check consistency
            is_consistent = explanation_sentiment == shap_sentiment
            feature_consistencies.append(is_consistent)
        
        if feature_consistencies:
            total_consistency += np.mean(feature_consistencies)
            valid_samples += 1
    
    return total_consistency / max(1, valid_samples)

def comprehensive_fidelity_metrics(explanations: List[str], 
                                 shap_dicts: List[Dict[str, float]],
                                 top_features_lists: List[List[str]]) -> Dict[str, float]:
    """
    Compute all fidelity metrics and return a comprehensive score.
    """
    metrics = {}
    
    # Original simple fidelity
    metrics['basic_fidelity'] = shap_fidelity_check(explanations, top_features_lists)
    
    # Advanced metrics
    metrics['ranked_fidelity'] = ranked_fidelity_score(explanations, shap_dicts)
    
    coverage_metrics = feature_coverage_fidelity(explanations, shap_dicts)
    metrics.update({f'coverage_{k}': v for k, v in coverage_metrics.items()})
    
    metrics['consistency_score'] = importance_consistency_score(explanations, shap_dicts)
    
    # Composite score (weighted average)
    weights = {
        'basic_fidelity': 0.2,
        'ranked_fidelity': 0.3,
        'coverage_f1': 0.3,
        'consistency_score': 0.2
    }
    
    composite_score = sum(weights[k] * metrics[k] for k in weights.keys() if k in metrics)
    metrics['composite_fidelity'] = composite_score
    
    return metrics
