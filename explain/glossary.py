import yaml
import os

# Placeholder. Replace with your glossary mapping (feature -> human-readable phrase).
# Example:
# GLOSSARY = {
#   "src_bytes": "bytes sent by the source",
#   "dst_bytes": "bytes received by the destination",
#   ...
# }
GLOSSARY = {}

def load_glossary_for_dataset(dataset: str) -> dict:
    """
    Load feature descriptions from the appropriate YAML file based on dataset.
    
    Args:
        dataset: 'nsl-kdd' or 'nb15'
    
    Returns:
        Dictionary mapping feature names to human-readable descriptions
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if dataset == 'nsl-kdd':
        features_file = os.path.join(current_dir, 'features_nsl_kdd.yaml')
    elif dataset == 'nb15':
        features_file = os.path.join(current_dir, 'features_nb15.yaml')
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'nsl-kdd' or 'nb15'")
    
    try:
        with open(features_file, 'r', encoding='utf-8') as f:
            glossary = yaml.safe_load(f)
        return glossary if glossary else {}
    except FileNotFoundError:
        print(f"Warning: Features file {features_file} not found. Using empty glossary.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading features file {features_file}: {e}. Using empty glossary.")
        return {}
