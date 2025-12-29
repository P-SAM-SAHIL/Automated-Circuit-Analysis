import json
import torch
from typing import List, Dict

from transformer_lens import HookedTransformer

from llm_client import RobustLLMClient
from config import MODEL_MAP
class TaskGenerator:
    """
    Generates task-specific counterfactual data for ACDC analysis.
    Replaces the hardcoded 'A B A B' patterns.
    """
    def __init__(self, client: RobustLLMClient, model: HookedTransformer):
        self.client = client
        self.model = model

    def generate_acdc_data(self, task_name: str, n_pairs: int = 5) -> List[Dict]:
        """
        Step 1 & 2: Synthesis and Formatting
        Generates Clean/Corrupt pairs for a specific task.
        """
        print(f"\n🎨 Synthesizing ACDC dataset for task: '{task_name}'...")

        # 1. THE PROMPT (MAIA-Style Synthesis)
        # We ask for pairs where ONLY the critical information changes.
        prompt = f"""
        I need to identify the circuit in a language model responsible for: "{task_name}".
        Generate {n_pairs} distinct pairs of (Clean, Corrupt) prompts to isolate this behavior via activation patching.

        RULES:
        1. "Clean": A prompt that triggers the desired behavior (e.g., correct prediction).
        2. "Corrupt": Almost identical to Clean, but change *one* key token so the prediction changes.
        3. Lengths MUST be identical (same number of tokens).

        Format example for 'Induction':
        {{ "clean": "The cat sat on the mat. The cat", "corrupt": "The cat sat on the mat. The dog" }}

        Return valid JSON only:
        {{
          "pairs": [
            {{ "clean": "...", "corrupt": "..." }},
            ...
          ]
        }}
        """

        # 2. CALL LLM
        try:
            response_str = self.client.chat(
                model=MODEL_MAP["hypothesis"], # Use a smart model here
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            data = json.loads(response_str)
            raw_pairs = data.get("pairs", [])
        except Exception as e:
            print(f"❌ Data generation failed: {e}")
            return []

        # 3. FORMAT FOR ACDC (The critical integration step)
        formatted_data = []
        
        print(f"   ✓ Generated {len(raw_pairs)} raw pairs. formatting...")

        for p in raw_pairs:
            try:
                clean_text = p['clean']
                corrupt_text = p['corrupt']

                clean_tokens = self.model.to_tokens(clean_text)
                corrupt_tokens = self.model.to_tokens(corrupt_text)

                # Ensure lengths match for patching
                if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
                    continue 

                # Run Clean Pass (Get Cache)
                with torch.no_grad():
                    _, clean_cache = self.model.run_with_cache(clean_tokens)
                
                # Run Corrupt Pass (Get Baseline Loss)
                with torch.no_grad():
                    corrupt_loss = self.model(corrupt_tokens, return_type="loss").item()

                # Structure exactly as your Interpreter expects
                formatted_data.append({
                    "clean_cache": clean_cache,
                    "corrupt_tokens": corrupt_tokens,
                    "corrupt_loss": corrupt_loss,
                    "clean_text": clean_text, # metadata
                    "corrupt_text": corrupt_text # metadata
                })
            except Exception as e:
                continue

        print(f"   ✓ Successfully formatted {len(formatted_data)} pairs for analysis.")
        return formatted_data