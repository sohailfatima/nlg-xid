from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from explain.llm_explainer import llm_explain
from explain.rules_nsl_kdd import explain_instance as explain_nsl
from explain.rules_nb15 import explain_instance as explain_nb15

def choose_rule_module(dataset: str):
    return explain_nsl if dataset == 'nsl-kdd' else explain_nb15

def build_explanations(dataset: str, preds: List[str], shap_frames: List[Dict[str, float]], instances: List[Dict[str, Any]], glossary: Dict[str,str], top_lists: List[List[str]], mode: str='rules', 
                       llm_provider='stub', llm_inputs='full', llm_variant='instruct', llm_model: str = None, api_key: str = None) -> List[str]:
    rule_func = choose_rule_module(dataset)
    outputs = []
    total = 0
    hits = 0
    for pred, shap_vals, inst, top in zip(preds, shap_frames, instances, top_lists):
        if mode == 'rules':
            results, hit = rule_func(inst, pred, shap_vals, glossary)
            hits += hit
            total += 1
            outputs.append(results)
        elif mode == 'llm':
            outputs.append(llm_explain(pred, top, shap_vals, inst, glossary, provider=llm_provider, variant=llm_variant, input_mode=llm_inputs, model=llm_model, api_key=api_key))
        elif mode == 'hybrid':
            # prepend rule snippet then ask LLM to elaborate
            rule_text = rule_func(inst, pred, shap_vals, glossary)
            llm_text = llm_explain(pred, top, shap_vals, inst, glossary, provider=llm_provider, variant=llm_variant, input_mode=llm_inputs, model=llm_model,api_key=api_key)
            outputs.append(rule_text + " " + llm_text)
        elif mode == 'shap_only':
            # Textualize SHAP without NLG
            parts = [f"{f}: {shap_vals.get(f,0):+.3f}" for f in top]
            outputs.append("Top factors -> " + "; ".join(parts))
        else:
            outputs.append("[Unknown explanation mode]")
    return outputs, (hits+0.0)/(total if total > 0 else 1)