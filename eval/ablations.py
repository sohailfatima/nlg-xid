from dataclasses import dataclass
from typing import Literal

@dataclass
class AblationConfig:
    model_type: Literal['xgb','ae'] = 'xgb'
    explanation_type: Literal['rules','llm','hybrid','shap_only'] = 'rules'
    llm_inputs: Literal['label','label+features','label+shap','full'] = 'full'
    llm_variant: Literal['instruct','uncensored'] = 'instruct'
