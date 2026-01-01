# ============================================================
# TASK GENERATOR (MECHANISM-SPECIFIC, ACDC-COMPATIBLE)
# ============================================================

import json
import torch
from typing import List, Dict

from transformer_lens import HookedTransformer
from llm_client import RobustLLMClient
from config import MODEL_MAP


class TaskGenerator:
    """
    Generates task-specific CLEAN / CORRUPT datasets for ACDC-style analysis.
    One task → one causal mechanism → one prompt template.
    """

    def __init__(self, client: RobustLLMClient, model: HookedTransformer):
        self.client = client
        self.model = model

    # ============================================================
    # PUBLIC ENTRY POINT (USED BY main.py)
    # ============================================================

    def generate_acdc_data(self, task_name: str, n_pairs: int = 5) -> List[Dict]:
        """
        Dispatches to a mechanism-specific prompt based on task_name.
        """
        task_lower = task_name.lower()

        if "induction" in task_lower:
            prompt = self._induction_prompt(n_pairs)

        elif "name" in task_lower or "indirect" in task_lower:
            prompt = self._name_mover_prompt(n_pairs)

        elif "position" in task_lower or "previous token" in task_lower:
            prompt = self._positional_prompt(n_pairs)

        else:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Supported: Induction, Name Mover (IOI), Positional."
            )

        print(f"\n🎨 Generating ACDC dataset for task: '{task_name}'")

        # ---------------------------
        # LLM CALL
        # ---------------------------
        try:
            response_str = self.client.chat(
                model=MODEL_MAP["hypothesis"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=1200
            )
            data = json.loads(response_str)
            raw_pairs = data.get("pairs", [])
        except Exception as e:
            print(f"❌ Data generation failed: {e}")
            return []

        print(f" ✓ Generated {len(raw_pairs)} raw pairs. Formatting...")

        # ---------------------------
        # FORMAT FOR ACDC
        # ---------------------------
        formatted_data = []

        for p in raw_pairs:
            try:
                clean_text = p["clean"]
                corrupt_text = p["corrupt"]

                clean_tokens = self.model.to_tokens(clean_text)
                corrupt_tokens = self.model.to_tokens(corrupt_text)

                # Critical ACDC constraint
                if clean_tokens.shape != corrupt_tokens.shape:
                    continue

                # Clean cache
# 1. Run Clean Pass (Get Cache AND Loss)
                with torch.no_grad():
                    # Calculate clean_loss as the "Perfect" baseline
                    clean_loss = self.model(clean_tokens, return_type="loss").item()
                    _, clean_cache = self.model.run_with_cache(clean_tokens)
                
                # 2. Run Corrupt Pass (Get Baseline Loss)
                with torch.no_grad():
                    corrupt_loss = self.model(corrupt_tokens, return_type="loss").item()

                # 3. Structure exactly as your Interpreter expects
                formatted_data.append({
                    "clean_cache": clean_cache,
                    "corrupt_tokens": corrupt_tokens,
                    "clean_loss": clean_loss,       # <--- THIS IS THE CRITICAL NEW FIELD
                    "corrupt_loss": corrupt_loss,
                    "clean_text": clean_text,
                    "corrupt_text": corrupt_text
                })

            except Exception:
                continue

        print(f" ✓ Successfully formatted {len(formatted_data)} ACDC pairs.")
        return formatted_data

    # ============================================================
    # PROMPT TEMPLATES (MECHANISM-SPECIFIC)
    # ============================================================

    def _induction_prompt(self, n_pairs: int) -> str:
        return f"""
    You are generating CLEAN / CORRUPT prompt pairs to isolate INDUCTION
    (token copying via repetition, A B ... A → B).

    Generate {n_pairs} pairs.

    CRITICAL RULES:
    1. CLEAN must contain a repeated token A followed by token B the first time it appears.
    2. The model should copy token B after the second occurrence of A.
    3. CORRUPT must break induction by changing ONLY the second occurrence of A.
    4. All tokens BEFORE the prediction position must be identical.
    5. Length and token positions MUST be identical.
    6. Prediction position is the final token.

    EXAMPLES:

    1)
    CLEAN:   "The cat sat on the mat . The cat sat"
    CORRUPT: "The cat sat on the mat . The dog sat"

    2)
    CLEAN:   "Alice went home early . Alice went"
    CORRUPT: "Alice went home early . Bob went"

    3)
    CLEAN:   "If you press the button , it starts . If you press"
    CORRUPT: "If you press the button , it starts . If you pull"

    4)
    CLEAN:   "John opened the door slowly . John opened"
    CORRUPT: "John opened the door slowly . Mary opened"

    Return valid JSON ONLY:
    {{
    "pairs": [
        {{ "clean": "...", "corrupt": "..." }}
    ]
    }}
    """


    def _name_mover_prompt(self, n_pairs: int) -> str:
        return f"""
    You are generating CLEAN / CORRUPT prompt pairs to isolate NAME-MOVER
    (Indirect Object Identification).

    Generate {n_pairs} pairs.

    CRITICAL RULES:
    1. Each sentence must contain EXACTLY two distinct names.
    2. Use a transfer verb (gave, sent, handed, showed).
    3. CLEAN: correct indirect object resolution.
    4. CORRUPT: swap the semantic roles of the two names.
    5. Token positions and sentence length MUST be identical.
    6. Prediction position is the final token.

     EXAMPLES:

    1)
    CLEAN:   "John gave Mary a book and then John thanked"
    CORRUPT: "Mary gave John a book and then John thanked"

    2)
    CLEAN:   "Alice sent Bob a message after Alice called"
    CORRUPT: "Bob sent Alice a message after Alice called"

    3)
    CLEAN:   "Tom handed Lisa the keys before Tom left"
    CORRUPT: "Lisa handed Tom the keys before Tom left"

    4)
    CLEAN:   "Sarah showed Mike the photo while Sarah smiled"
    CORRUPT: "Mike showed Sarah the photo while Sarah smiled"

    Return valid JSON ONLY:
    {{
    "pairs": [
        {{ "clean": "...", "corrupt": "..." }}
    ]
    }}
    """


    def _positional_prompt(self, n_pairs: int) -> str:
        return f"""


You are generating CLEAN / CORRUPT prompt pairs to isolate
PREVIOUS-TOKEN ATTENTION
(attending to the immediately preceding token).

Generate {n_pairs} pairs.

CRITICAL RULES (DO NOT VIOLATE):

1. The final token in the sequence MUST be predicted by the model.
2. The correct prediction MUST depend ONLY on the IMMEDIATELY PREVIOUS token.
3. CLEAN and CORRUPT must:
   - have IDENTICAL length
   - use the SAME vocabulary
   - differ ONLY in the token at position i−1
4. Tokens must NOT repeat.
5. All tokens BEFORE the final two positions must be identical.

EXAMPLES:

1)
CLEAN:   "red blue green yellow orange"
CORRUPT: "red blue green yellow purple"

2)
CLEAN:   "one two three four five"
CORRUPT: "one two three four six"

3)
CLEAN:   "alpha beta gamma delta epsilon"
CORRUPT: "alpha beta gamma delta zeta"

IMPORTANT:
- The model predicts the FINAL token.
- The ONLY difference is the token at position i−1.
- The prediction must change if and only if attention to i−1 changes.

Return valid JSON ONLY:
{{
  "pairs": [
    {{ "clean": "...", "corrupt": "..." }}
  ]
}}


    """
