import json
import numpy as np
from typing import List, Dict
import openai
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import torch
from core.hypothesis import CircuitHypothesis
from config import MODEL_MAP
from llm_client import RobustLLMClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score



from acdc.TLACDCExperiment import TLACDCExperiment
import torch.nn.functional as F
from acdc.acdc_utils import kl_divergence



class AdvancedAutomatedInterpreter:
    """
    LLM-driven circuit discovery with:
    - Attribution graph interpretation [ACDC-style]
    - Circuit-level reasoning
    - Automated probe design
    - OOD/adversarial test generation
    """

    def __init__(self, model: HookedTransformer, client: RobustLLMClient):
            # SIMPLIFIED INIT: No dataframe, no hardcoded data pairs
            self.model = model
            self.client = client
            self.edge_matrix = None
            self.hypotheses = []
            self.current_graph_edges = [] # Store raw edges for graph analysis
    # ============================================================
    # PART 1: ATTRIBUTION GRAPH EXTRACTION (ACDC-style)
    # ============================================================

    def build_attribution_graph(self, data: List[Dict], threshold: float = 0.05):
            print(f"Building Attribution Graph (Threshold: {threshold})...")
            
            # 1. Handle Variable Lengths via Left Padding
            # : Visualizing [PAD, PAD, T1, T2] alignment
            max_len = max([p["clean_tokens"].shape[-1] for p in data])
            pad_token = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id else 50256

            def pad_seq(seq, max_l, pad_token):
                curr_l = seq.shape[-1]
                if curr_l == max_l: return seq
                return F.pad(seq, (max_l - curr_l, 0), "constant", pad_token)

            clean_list = [pad_seq(p["clean_tokens"], max_len, pad_token).squeeze(0) for p in data]
            corrupt_list = [pad_seq(p["corrupt_tokens"], max_len, pad_token).squeeze(0) for p in data]

            # FIX: Force Long (int64) type. F.pad might have altered dtype or layout.
            clean_data = torch.stack(clean_list).long()
            corrupt_data = torch.stack(corrupt_list).long()

            # DEBUG: Verify tensor properties
            print(f"DEBUG: Data Shape: {clean_data.shape}, Dtype: {clean_data.dtype}, Device: {clean_data.device}")

                # 2. DEFINE ACDC METRIC (KL Divergence) - CRITICAL FIX APPLIED
                # We perform the slice [:, -1, :] here to ensure shapes match ACDC default metric
            with torch.no_grad():
                
                clean_logits = self.model(clean_data, prepend_bos=False)
                clean_log_probs = F.log_softmax(clean_logits[:, -1, :], dim=-1)

            def acdc_metric(logits):
                return kl_divergence(logits, clean_log_probs)

            # 3. INITIALIZE EXPERIMENT
            self.model.reset_hooks()
            exp = TLACDCExperiment(
                model=self.model,
                threshold=threshold,
                using_wandb=False,
                zero_ablation=False, 
                ds=clean_data,
                ref_ds=corrupt_data,
                metric=acdc_metric,
                verbose=True,
                indices_mode="reverse", 
                names_mode="normal",

                corrupted_cache_cpu=False,
                online_cache_cpu=False,
                add_sender_hooks=True,
                add_receiver_hooks=False,
                remove_redundant=False,
                show_full_index=False,

            )
            # 4. RUN RECURSIVE PRUNING
            print("Starting recursive pruning...")
            MAX_STEPS = 10000 
            step_count = 0
            
            try:
                while exp.current_node is not None and step_count < MAX_STEPS:
                    exp.step(testing=False)
                    step_count += 1
            except Exception as e:
                print(f"ACDC Loop ended with error (or finished): {e}")

            # 5. PARSE GRAPH (The Fix for MLP Support)
            # We clear the previous edges list to ensure no stale state
            self.current_graph_edges = []  
            
            count = 0

            # Iterate through ALL receivers (Attn Heads AND MLPs) in the ACDC correspondence graph
            for receiver_name, receiver_indices in exp.corr.edges.items():
                
                # 1. Parse Receiver Layer
                try:
                    r_parts = receiver_name.split(".")
                    r_layer = int(r_parts[1])
                except:
                    continue # Skip non-standard names (like embedding/blocks.0.hook_resid_pre)

                # 2. Identify Receiver Type (Head vs MLP)
                r_type = "MLP" if "mlp" in receiver_name else "Head"
                r_head_idx = -1 # Default for MLP (effectively "None")
                
                # Case A: Receiver is a Head
                if r_type == "Head":
                    for receiver_index, senders in receiver_indices.items():
                        try:
                            # ACDC TorchIndex usually: (None, None, head_idx)
                            if len(receiver_index.hashable_tuple) >= 3:
                                r_head_idx = receiver_index.hashable_tuple[2]
                            
                            if r_head_idx is None: continue 
                            
                            # Process Senders for this specific Head
                            edge_count = self._process_senders(senders, r_layer, r_head_idx, "Head")
                            count += edge_count
                        except:
                            continue
                
                # Case B: Receiver is an MLP
                elif r_type == "MLP":
                    # MLP acts on the whole residual stream, so index is usually just [None]
                    for receiver_index, senders in receiver_indices.items():
                        edge_count = self._process_senders(senders, r_layer, -1, "MLP")
                        count += edge_count

            self.model.reset_hooks()
            print(f"ACDC complete. Found {count} active edges (including MLPs).")
            return self.current_graph_edges

    def _process_senders(self, senders, r_layer, r_head, r_type):
            local_count = 0
            for sender_name, sender_indices in senders.items():
                for sender_index, edge in sender_indices.items():
                    if not edge.present: continue

                    # --- FIX START ---
                    s_head_idx = -1
                    s_type = "Head"
                    
                    # Handle Special Input Nodes
                    if "embed" in sender_name or "resid_pre" in sender_name:
                        s_layer = 0
                        s_type = "Input"
                    else:
                        try:
                            s_parts = sender_name.split(".")
                            s_layer = int(s_parts[1])
                        except:
                            continue # Actually skip if really unparseable
                    
                    if "mlp" in sender_name:
                        s_type = "MLP"
                    # --- FIX END ---

                    if s_type == "Head":
                        # Check for head index in TorchIndex tuple
                        if len(sender_index.hashable_tuple) >= 3:
                            s_head_idx = sender_index.hashable_tuple[2]

                    self.current_graph_edges.append({
                        'src_layer': s_layer,
                        'src_head': s_head_idx,
                        'src_type': s_type,
                        'dst_layer': r_layer,
                        'dst_head': r_head,
                        'dst_type': r_type,
                        'effect_size': edge.effect_size
                    })
                    local_count += 1
            return local_count

# ... inside AdvancedAutomatedInterpreter class ...

    def format_graph_for_llm(self) -> str:
            if not self.current_graph_edges:
                return "No significant connections found."

            graph_text = "Attribution Graph (Causal Connections):\n"
            
            # Sort edges by layer flow
            sorted_edges = sorted(
                self.current_graph_edges, 
                key=lambda x: (x['dst_layer'], x['dst_head'], x['src_layer'])
            )
            
            lines = []
            for edge in sorted_edges:
                # Format Source Name
                if edge['src_type'] == 'MLP':
                    src_name = f"L{edge['src_layer']}.MLP"
                else:
                    src_name = f"L{edge['src_layer']}.H{edge['src_head']}"

                # Format Dest Name
                if edge['dst_type'] == 'MLP':
                    dst_name = f"L{edge['dst_layer']}.MLP"
                else:
                    dst_name = f"L{edge['dst_layer']}.H{edge['dst_head']}"

                lines.append(f"  {src_name} → {dst_name}")

            graph_text += "\n".join(lines)
            return graph_text

    # ============================================================
    # STEP 3: GRAPH -> HIGH-LEVEL CAUSAL INTERPRETATION (LLM)
    # ============================================================

    def generate_circuit_hypotheses(self, task_description: str, 
                                    max_hypotheses: int = 1) -> List[CircuitHypothesis]:
        """
        Step 3: Interprets the WHOLE Graph for the given task.
        Input: 
            - task_description (from Step 1)
            - attribution_graph (internal state from Step 2)
        """
        
        # 1. Get the full graph text (no filtering)
        graph_text = self.format_graph_for_llm()

        context = f"""
You are a MECHANISTIC INTERPRETABILITY RESEARCHER.

TASK DESCRIPTION:
"{task_description}"

EVIDENCE (CAUSAL ATTRIBUTION GRAPH):
The following edges were found to be CAUSALLY NECESSARY by ACDC (Automated Circuit Discovery):
{graph_text}

YOUR GOAL:
Synthesize these edges into a single, coherent ALGORITHMIC MECHANISM.
Identify the "Circuit Path" (the chain of heads) that performs the task.

IMPORTANT CONSTRAINTS:
1. Rely ONLY on the provided graph and task description.
2. Do not hallucinate heads not present in the graph.
3. The mechanism must explain *how* the graph solves the *task*.

Examples of Mechanisms:
- "L0.H1 attends to previous token → L1.H4 copies content → Output"
- "L10.H7 inhibits L11.H10 to prevent incorrect name prediction"

Return VALID JSON ONLY:
{{
  "hypotheses": [
    {{
      "circuit_path": [[0, 1], [1, 4]], 
      "mechanism": "Precise description of algorithm...",
      "predicted_behavior": "Falsifiable prediction...",
      "test_strategy": "last"
    }}
  ]
}}
"""

        try:
            response_content = self.client.chat(
                model=MODEL_MAP["hypothesis"],
                messages=[{"role": "user", "content": context}],
                temperature=0.7,
                max_tokens=1000
            )

            response_data = json.loads(response_content)
            hypotheses_data = response_data.get("hypotheses", [])

            new_hypotheses = []
            for h in hypotheses_data[:max_hypotheses]:
                path_list = h.get('circuit_path', [])
                parsed_path = [(int(p[0]), int(p[1])) for p in path_list]
                
                # Since we analyze the whole graph, we don't have a single "target head" 
                # effectively. We set it to the last head in the chain or -1.
                t_layer = parsed_path[-1][0] if parsed_path else -1
                t_head = parsed_path[-1][1] if parsed_path else -1

                new_hypotheses.append(CircuitHypothesis(
                    target_layer=t_layer,
                    target_head=t_head,
                    circuit_path=parsed_path,
                    mechanism=h.get('mechanism', 'Unknown mechanism'),
                    predicted_behavior=h.get('predicted_behavior', 'Unknown behavior')
                ))

            self.hypotheses.extend(new_hypotheses)
            print(f"  ✓ Generated {len(new_hypotheses)} causal hypotheses based on full graph.")
            return new_hypotheses

        except Exception as e:
            print(f"  Hypothesis generation failed: {e}")
            return []

    # ============================================================
    # PART 3: AUTOMATED PROBE DESIGN (FIXED)
    # ============================================================

# ============================================================
    # STEP 4: AUTOMATED VALIDATION
    # ============================================================

# ============================================================
    # STEP 4: AUTOMATED VALIDATION (OPTIMIZED PROBE)
    # ============================================================

    def design_and_test_probe(self, hypothesis: CircuitHypothesis, n_samples: int = 40) -> float:
        """
        4.1 Probe Generation & Testing (Batched & Optimized)
        Generates data -> Extracts activations -> Trains Logistic Regression.
        """
        print(f"  Designing probe for mechanism: {hypothesis.mechanism[:50]}...")

        # 1. GENERATE PROBE DATASET (LLM)
        # We ask for a balanced dataset of 'active' (feature present) and 'inactive' (feature absent)
        probe_prompt = f"""
You are designing a LINEAR PROBE DATASET to validate a mechanistic hypothesis.

HYPOTHESIS: "{hypothesis.mechanism}"
BEHAVIOR: "{hypothesis.predicted_behavior}"

Task: Generate {n_samples} pairs of short text examples.
- "active": The mechanism SHOULD fire (feature is present).
- "inactive": The mechanism SHOULD NOT fire (feature is absent).

Constraint: The "active" and "inactive" examples should be as similar as possible (minimal pairs).
Do NOT include explanations.

Return VALID JSON ONLY:
{{
  "examples": [
    {{ "text": "The cat sat on the mat", "label": "active" }},
    {{ "text": "The dog ran on the grass", "label": "inactive" }}
  ]
}}
"""
        try:
            response_content = self.client.chat(
                model=MODEL_MAP["probe"],
                messages=[{"role": "user", "content": probe_prompt}],
                temperature=0.8,
                max_tokens=2500
            )
            dataset = json.loads(response_content).get("examples", [])
        except Exception as e:
            print(f"    Probe generation failed: {e}")
            return 0.0

        if not dataset or len(dataset) < 10:
            print("    Not enough data generated for probing.")
            return 0.0

        # 2. PREPARE BATCH DATA
        texts = [item['text'] for item in dataset]
        labels = [1 if item['label'] == 'active' else 0 for item in dataset]
        
        # 3. IDENTIFY TARGET COMPONENT
        # If target is -1 (MLP or general), we default to the last component in the circuit path
        target_layer, target_head = hypothesis.target_layer, hypothesis.target_head
        
        if target_layer == -1:
             if hypothesis.circuit_path:
                 target_layer, target_head = hypothesis.circuit_path[-1]
             else:
                 return 0.0

        # Determine Hook Name and Extraction Logic
        # If target_head is -1, it's an MLP or residual stream -> [batch, seq, d_model]
        # If target_head is >= 0, it's a specific Head -> [batch, seq, d_head]
        if target_head == -1:
            hook_name = get_act_name("mlp_out", target_layer) # Or "resid_post" depending on hypothesis
            use_head_idx = False
        else:
            hook_name = get_act_name("z", target_layer) # Output of attention heads
            use_head_idx = True

        # 4. BATCH ACTIVATION EXTRACTION (Speedup)
        try:
            # Tokenize all at once
            tokens = self.model.to_tokens(texts, prepend_bos=True)
            
            # Run with cache - only caching the specific hook we need to save memory
            _, cache = self.model.run_with_cache(
                tokens, 
                names_filter=lambda x: x == hook_name
            )
            
            # Extract activations at the last token position
            # Shape of acts: [batch, seq_len, ...]
            raw_acts = cache[hook_name] 
            
            # Take the last token's activation (prediction position)
            # Note: If your model uses left-padding, -1 is correct. 
            # If right-padding, you'd need the sequence lengths. TransformerLens usually handles this.
            final_token_acts = raw_acts[:, -1, :] 
            
            if use_head_idx:
                # Shape: [batch, n_heads, d_head] -> select specific head -> [batch, d_head]
                X = final_token_acts[:, target_head, :].float().cpu().numpy()
            else:
                # Shape: [batch, d_model]
                X = final_token_acts.float().cpu().numpy()
                
            y = np.array(labels)

        except Exception as e:
            print(f"    Error extracting activations: {e}")
            return 0.0

        # 5. TRAIN PROBE (Scikit-Learn)
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            # Use a pipeline to scale data (important for convergence)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='liblinear'))
            
            # 3-Fold Cross Validation for robustness
            if len(y) < 5: return 0.0 # Safety check
            
            scores = cross_val_score(clf, X, y, cv=min(3, len(y)//2))
            accuracy = scores.mean()
            
            print(f"    ✓ Probe Accuracy: {accuracy:.2f} (Target: L{target_layer}.{'MLP' if target_head==-1 else 'H'+str(target_head)})")
            return float(accuracy)

        except ImportError:
            print("    sklearn not installed. Please run `pip install scikit-learn`.")
            return 0.0
        except Exception as e:
            print(f"    Probe training error: {e}")
            return 0.0

    def generate_adversarial_prompts(self, hypothesis: CircuitHypothesis, n_prompts: int = 5) -> List[str]:
        """
        4.3 OOD / Adversarial Inputs
        LLM generates inputs specifically designed to BREAK the mechanism.
        """
        prompt = f"""
You are generating ADVERSARIAL TEST PROMPTS for MECHANISTIC VALIDATION.

TARGET MECHANISM:
"{hypothesis.mechanism}"

PREDICTED BEHAVIOR:
"{hypothesis.predicted_behavior}"

GOAL:
Generate prompts that are:
- minimally different from normal cases
- specifically designed to BREAK this mechanism if it is real

STRICT RULES:
1. Prompts must LOOK natural (no random symbols or emojis).
2. Each prompt should remove or weaken the exact signal used by the mechanism.
3. Do NOT destroy overall sentence coherence.
4. Do NOT introduce unrelated noise.

MECHANISM-SPECIFIC GUIDANCE:

If the mechanism involves INDUCTION:
- Break repetition (A B ... A → ?)
- Replace the second A with a near synonym or different token
- Keep all other structure identical

If the mechanism involves NAME-MOVER:
- Add distractor clauses
- Introduce role ambiguity
- Swap semantic roles without changing surface order

If the mechanism involves POSITION:
- Insert filler tokens
- Shift token positions subtly
- Preserve vocabulary but change offsets

GOOD EXAMPLES:

Induction:
- "The cat sat down . The dog sat"
- "Alice went home early . Bob went"

Name-mover:
- "John gave Mary a book and Mary thanked"
- "Alice sent Bob a letter before Bob replied"

Positional:
- "red blue green yellow"
- "blue red green yellow"

Generate exactly {n_prompts} prompts.

Return VALID JSON ONLY:
{{
  "prompts": ["...", "...", "..."]
}}
"""
        try:
            response = self.client.chat(
                model=MODEL_MAP["adversarial"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )
            return json.loads(response).get("prompts", [])[:n_prompts]
        except Exception:
            return ["The quick brown fox jumps over the lazy dog"] # Fallback

    def execute_intervention(self, hypothesis: CircuitHypothesis, prompts: List[str]) -> float:
        """
        4.2 Intervention Tests (Ablation)
        We ablate the ENTIRE hypothesized circuit path and measure impact.
        Returns: Average relative change in loss (Impact Score).
        """
        if not hypothesis.circuit_path or not prompts:
            return 0.0

        print(f"  Running ablation intervention on {len(prompts)} prompts...")
        
        # Organize heads by layer for efficient hooking
        heads_to_ablate = {}
        for l, h in hypothesis.circuit_path:
            heads_to_ablate.setdefault(l, []).append(h)

        impact_scores = []

        for prompt in prompts:
            try:
                tokens = self.model.to_tokens(prompt)
                
                # 1. Baseline Loss
                clean_loss = self.model(tokens, return_type="loss").item()
                
                # 2. Ablated Loss (Zero-ablate specific heads)
                def ablate_hook(z, hook):
                    layer = hook.layer()
                    if layer in heads_to_ablate:
                        for h in heads_to_ablate[layer]:
                            z[:, :, h, :] = 0.0 # Zero ablation
                    return z
                
                hooks = [(get_act_name("z", l), ablate_hook) for l in heads_to_ablate.keys()]
                
                ablated_loss = self.model.run_with_hooks(
                    tokens, 
                    return_type="loss", 
                    fwd_hooks=hooks
                ).item()

                # Score: How much did loss INCREASE? (Normalized)
                # If loss goes 3.0 -> 6.0, score is 1.0 (100% increase)
                if clean_loss > 0:
                    pct_change = (ablated_loss - clean_loss) / clean_loss
                    impact_scores.append(pct_change)
            except Exception:
                continue

        avg_impact = float(np.mean(impact_scores)) if impact_scores else 0.0
        print(f"    ✓ Intervention Impact (Avg Loss Incr): {avg_impact:.2%}")
        return avg_impact

    def validate_circuit_hypothesis(self, hypothesis: CircuitHypothesis) -> Dict:
        """
        4.4 Verdict Layer
        Aggregates validation signals into a final verdict.
        NO human review involved.
        """
        print(f"\n🔬 AUTOMATED VALIDATION: {hypothesis.mechanism[:40]}...")

        # 1. Run Probe (Does the head represent the feature?)
        probe_acc = self.design_and_test_probe(hypothesis)
        
        # 2. Generate OOD Prompts for Interventions
        adv_prompts = self.generate_adversarial_prompts(hypothesis)
        
        # 3. Run Interventions (Is the circuit necessary?)
        # We test on OOD prompts to see if the circuit matters there
        intervention_effect = self.execute_intervention(hypothesis, adv_prompts)

        # 4. Scoring Logic (Adjust weights as needed)
        # Probe > 0.7 is good. Intervention > 0.1 (10% loss increase) is significant.
        
        score_probe = max(0, (probe_acc - 0.5) * 2) # Normalize 0.5-1.0 to 0.0-1.0
        score_intervention = min(1.0, intervention_effect * 5) # Cap effect at 20% loss increase = 1.0
        
        # Simple weighted sum
        overall_score = (score_probe * 0.6) + (score_intervention * 0.4)
        
        hypothesis.confidence = overall_score
        
        verdict = "REJECTED"
        if overall_score > 0.7: verdict = "SUPPORTED"
        elif overall_score > 0.4: verdict = "WEAK"

        result = {
            'hypothesis': hypothesis.mechanism,
            'circuit_path': hypothesis.circuit_path,
            'probe_accuracy': probe_acc,
            'intervention_effect': intervention_effect,
            'overall_score': overall_score,
            'verdict': verdict
        }
        
        print(f"  🏁 VERDICT: {verdict} (Score: {overall_score:.2f})")
        return result