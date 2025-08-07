from typing import Dict, Any, List
import textwrap
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PROMPT_TEMPLATE = """You are a cybersecurity expert analyzing network traffic for a Security Operations Center (SOC).

Task: Explain why the machine learning model classified this network event as "{prediction}".

Key Contributing Features (SHAP values show feature importance):
{top_features}

Event Details:
{raw_values}

Instructions:
- Provide a clear, technical explanation suitable for SOC analysts
- Focus on the most important features and their cybersecurity implications
- Explain how these features support the classification decision
- Use precise cybersecurity terminology
- Keep the explanation concise but comprehensive
- Do not contradict the evidence provided

Explanation:"""

def render_prompt(prediction: str, shap_top_list: List[str], shap_values: Dict[str, float], instance: Dict[str, Any], glossary: Dict[str, str], variant: str='instruct', input_mode: str='full'):
    # input_mode ablation: 'label', 'label+features', 'label+shap', 'full'
    s = lambda f: glossary.get(f, f)
    if input_mode == 'label':
        top_str = "N/A"
        raw_str = "N/A"
    elif input_mode == 'label+features':
        top_str = ", ".join(s(f) for f in shap_top_list)
        raw_str = "N/A"
    elif input_mode == 'label+shap':
        top_str = ", ".join(f"{s(f)} ({shap_values.get(f,0):+.3f})" for f in shap_top_list)
        raw_str = "N/A"
    else:
        top_str = ", ".join(f"{s(f)} ({shap_values.get(f,0):+.3f})" for f in shap_top_list)
        raw_str = ", ".join(f"{s(k)}={v}" for k,v in list(instance.items())[:20])  # truncate for readability
    return PROMPT_TEMPLATE.format(prediction=prediction, top_features=top_str, raw_values=raw_str)

class OllamaClient:
    """Client for Ollama-hosted Llama models"""
    
    def __init__(self, model: str = 'llama2', base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url
        self.provider = 'ollama'
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text using Ollama API"""
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["\n\n", "Human:", "Assistant:"]
                }
            }
            
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama request failed: {e}")
            return self._fallback_explanation(prompt)
        except Exception as e:
            print(f"Ollama generation failed: {e}")
            return self._fallback_explanation(prompt)
    
    def _fallback_explanation(self, prompt: str) -> str:
        """Fallback explanation when Ollama fails"""
        return "Network event classified based on feature analysis. Ollama service unavailable for detailed explanation."


class HuggingFaceClient:
    """Client for Hugging Face Transformers Llama models"""
    
    def __init__(self, model_name: str = 'meta-llama/Llama-2-7b-chat-hf'):
        self.model_name = model_name
        self.provider = 'huggingface'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                load_in_8bit=True if torch.cuda.is_available() else False
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"Model loaded on {self.device}")
            
        except Exception as e:
            print(f"Failed to load Hugging Face model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text using Hugging Face model"""
        if self.model is None or self.tokenizer is None:
            return self._fallback_explanation(prompt)
        
        try:
            # Format prompt for Llama-2-chat
            if 'chat' in self.model_name.lower():
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"Hugging Face generation failed: {e}")
            return self._fallback_explanation(prompt)
    
    def _fallback_explanation(self, prompt: str) -> str:
        """Fallback explanation when model fails"""
        return "Network event classified based on feature analysis. Hugging Face model unavailable for detailed explanation."


class OpenRouterClient:
    """Client for OpenRouter API"""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-maverick:free", base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.provider = 'openrouter'

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text using OpenRouter API"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/",
            "X-Title": "llm_explainer"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"OpenRouter generation failed: {e}")
            return self._fallback_explanation(prompt)

    def _fallback_explanation(self, prompt: str) -> str:
        """Fallback explanation when OpenRouter fails"""
        return "Network event classified based on feature analysis. OpenRouter service unavailable for detailed explanation."


    

class LLMClientStub:
    """Fallback stub client"""
    
    def __init__(self, provider: str='stub', model: str='stub-001'):
        self.provider = provider
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        # Fallback: produce a compact explanation using the prompt contents
        return textwrap.shorten("Based on the given features and their SHAP impacts, the classification aligns with the highlighted network behaviors. " + prompt, width=600)

def create_llm_client(provider: str, model: str = None, api_key: str = None):
    """Factory function to create appropriate LLM client"""
    if provider == 'ollama':
        model = model or 'llama2'
        return OllamaClient(model=model)
    elif provider == 'huggingface':
        model = model or 'meta-llama/Llama-2-7b-chat-hf'
        return HuggingFaceClient(model_name=model)
    elif provider == 'openrouter':
        if api_key is None:
            raise ValueError("API key required for OpenRouter provider")
        model = model or "meta-llama/llama-4-maverick:free"
        return OpenRouterClient(api_key=api_key, model=model)
    else:
        return LLMClientStub(provider=provider, model=model or 'stub-001')


def llm_explain(prediction: str, shap_top_list: List[str], shap_values: Dict[str, float], 
                instance: Dict[str, Any], glossary: Dict[str, str], provider: str='stub', 
                variant: str='instruct', input_mode: str='full', client=None, 
                model: str = None, api_key: str = None) -> str:
    """
    Generate LLM-based explanation for model predictions
    
    Args:
        prediction: The model's prediction/classification
        shap_top_list: List of top contributing features
        shap_values: Dictionary of feature names to SHAP values
        instance: Dictionary of feature names to values for this instance
        glossary: Dictionary mapping feature names to human-readable descriptions
        provider: LLM provider ('ollama', 'huggingface', 'stub')
        variant: Prompt variant (currently unused, for future extensibility)
        input_mode: What information to include ('label', 'label+features', 'label+shap', 'full')
        client: Pre-initialized client (optional)
        model: Specific model name to use
    
    Returns:
        Generated explanation string
    """
    # Create client if not provided
    if client is None:
        client = create_llm_client(provider, model,api_key)
    
    # Generate the prompt
    prompt = render_prompt(prediction, shap_top_list, shap_values, instance, glossary, variant, input_mode)
    
    # Generate explanation
    try:
        explanation = client.generate(prompt)
        
        # Post-process the explanation
        explanation = explanation.strip()
        
        # Remove any remaining prompt artifacts
        if explanation.lower().startswith('explanation:'):
            explanation = explanation[12:].strip()
        
        # Ensure it's not too long
        if len(explanation) > 1000:
            explanation = textwrap.shorten(explanation, width=1000, placeholder="...")
        
        return explanation
        
    except Exception as e:
        print(f"LLM explanation failed: {e}")
        return f"Classification: {prediction}. Key features: {', '.join(shap_top_list[:3])}. Detailed analysis unavailable."
