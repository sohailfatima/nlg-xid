import numpy as np
import shap
import pandas as pd

def _aggregate_shap_to_original_features(shap_values, feature_names_out, original_feature_names):
    """
    Aggregate SHAP values from transformed features back to original features.
    Handles one-hot encoded categorical features by summing their SHAP values.
    """
    # Create a mapping from transformed features to original features
    feature_to_original = {}
    
    for i, transformed_name in enumerate(feature_names_out):
        if transformed_name.startswith('cat__'):
            # For categorical features, extract the original feature name
            # Format: cat__original_feature_value
            parts = transformed_name.split('__', 1)
            if len(parts) > 1:
                # Find the original feature name by matching with original_feature_names
                for orig_name in original_feature_names:
                    if parts[1].startswith(orig_name):
                        feature_to_original[i] = orig_name
                        break
                else:
                    # Fallback: use the part after cat__
                    base_name = parts[1].split('_')[0]  # Take first part before underscore
                    feature_to_original[i] = base_name
        elif transformed_name.startswith('num__'):
            # For numerical features, direct mapping
            parts = transformed_name.split('__', 1)
            if len(parts) > 1:
                feature_to_original[i] = parts[1]
        else:
            # Fallback: use as is
            feature_to_original[i] = transformed_name
    
    # Aggregate SHAP values by original feature
    aggregated_shap = {}
    for sample_idx in range(shap_values.shape[0]):
        sample_dict = {}
        for feat_idx, orig_name in feature_to_original.items():
            if orig_name in sample_dict:
                sample_dict[orig_name] += shap_values[sample_idx, feat_idx]
            else:
                sample_dict[orig_name] = shap_values[sample_idx, feat_idx]
        aggregated_shap[sample_idx] = sample_dict
    
    # Convert to DataFrame
    df = pd.DataFrame([aggregated_shap[i] for i in range(shap_values.shape[0])])
    return df


def shap_for_xgb(trained_pipeline, X_sample, background_samples=200, top_k=5):
    # trained_pipeline: Pipeline(preprocessor='pre', classifier='clf')
    pre = trained_pipeline.named_steps['pre']
    clf = trained_pipeline.named_steps['clf']

    # Transform and ensure dense numeric array
    X_trans = pre.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    X_trans = np.asarray(X_trans, dtype=np.float32)

    # Build explainer and compute SHAP
    explainer = shap.TreeExplainer(clf)

    # Newer SHAP returns an Explanation object from calling the explainer
    sv = explainer(X_trans)

    # Normalize to a 2-D (n_samples, n_features) array of magnitudes
    if hasattr(sv, "values"):  # Explanation
        vals = np.abs(sv.values)
        if vals.ndim == 3:      # (n, d, k) multiclass
            vals = vals.mean(axis=2)
    else:
        # Older API: sv could be np.ndarray or list[np.ndarray]
        if isinstance(sv, list):  # list of (n, d) per class
            vals = np.mean([np.abs(v) for v in sv], axis=0)
        else:  # (n, d)
            vals = np.abs(sv)

    feature_names_out = pre.get_feature_names_out()
    original_feature_names = list(X_sample.columns)

    # Sanity check
    assert vals.shape[1] == len(feature_names_out), \
        f"SHAP feature dim {vals.shape[1]} != names {len(feature_names_out)}"

    # Aggregate to original features
    df = _aggregate_shap_to_original_features(vals, feature_names_out, original_feature_names)
    df.index = X_sample.index

    # Optional: drop label columns if present
    df = df.drop(columns=['label', 'attack_cat'], errors='ignore')

    # Top-K original features by mean |SHAP|
    top = df.abs().mean(0).sort_values(ascending=False).head(top_k)
    return df, list(top.index)

def shap_for_scorer(trained_pipeline, scorer_fn, X_sample, background_samples=200, top_k=5, nsamples_kernel=200):
    # Computes SHAP for an arbitrary scorer, e.g., reconstruction error from AE.
    pre = trained_pipeline.named_steps['pre']
    X_trans = pre.transform(X_sample)
    # Background for KernelExplainer
    bg_idx = np.random.choice(X_trans.shape[0], size=min(background_samples, X_trans.shape[0]), replace=False)
    background = X_trans[bg_idx]
    f = lambda x: scorer_fn(x)  # expects x in transformed space
    explainer = shap.KernelExplainer(f, background)
    shap_vals = explainer.shap_values(X_trans, nsamples=nsamples_kernel)
    
    # Get feature names from preprocessor and original feature names
    feature_names_out = pre.get_feature_names_out()
    original_feature_names = list(X_sample.columns)
    
    # Aggregate SHAP values back to original features
    df = _aggregate_shap_to_original_features(shap_vals, feature_names_out, original_feature_names)
    df.index = X_sample.index
    
    # Drop label columns from the final DataFrame if they exist
    df = df.drop(columns=['label', 'attack_cat'], errors='ignore')
    
    top = df.abs().mean(0).sort_values(ascending=False).head(top_k)
    return df, list(top.index)
