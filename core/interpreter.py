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

from utils.cache_utils import safe_get_z
from utils.patching import patch_head_delta



class AdvancedAutomatedInterpreter:
    """
    LLM-driven circuit discovery with:
    - Attribution graph interpretation [ACDC-style]
    - Circuit-level reasoning
    - Automated probe design
    - OOD/adversarial test generation
    """

    def __init__(self, model, df_results, pair_data_induction, pair_data_name_mover,client):
        self.model = model
        self.df = df_results
        self.induction_pairs = pair_data_induction
        self.nm_pairs = pair_data_name_mover
        self.hypotheses = []
        self.edge_matrix = None
        self.circuit_cache = {}
        self.client=client

    # ============================================================
    # PART 1: ATTRIBUTION GRAPH EXTRACTION (ACDC-style)
    # ============================================================

    def build_attribution_graph(self, top_k_heads: int = 30, threshold: float = 0.05):
        """Extract attribution graph showing head→head importance"""
        print("\n Building attribution graph...")

        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        n_total = n_layers * n_heads

        self.edge_matrix = np.zeros((n_total, n_total))

        top_heads = self.df.nlargest(top_k_heads, 'induction_score')

        for idx1, row1 in top_heads.iterrows():
            for idx2, row2 in top_heads.iterrows():
                l1, h1 = int(row1['layer']), int(row1['head'])
                l2, h2 = int(row2['layer']), int(row2['head'])

                if l1 < l2:
                    importance = self._compute_edge_importance(l1, h1, l2, h2)

                    if importance > threshold:
                        flat_idx1 = l1 * n_heads + h1
                        flat_idx2 = l2 * n_heads + h2
                        self.edge_matrix[flat_idx1, flat_idx2] = importance

        print(f"✓ Found {np.sum(self.edge_matrix > 0)} significant edges")
        return self.edge_matrix


    def _compute_edge_importance(
        self,
        layer1: int, head1: int,
        layer2: int, head2: int
    ) -> float:
        """
        ACDC-compliant Edge Importance via Path Patching.
        
        Measures the causal contribution of the edge:
            (layer1.head1) → (layer2.head2)
        
        by patching the signal through Q, K, and V inputs of the receiver head.
        
        References:
        - Conmy et al. (2023): "Towards Automated Circuit Discovery"
        - ACDC Algorithm: Path patching through residual stream
        
        Implementation:
        1. Compute H1's output difference (clean - corrupt)
        2. Project through W_O to get residual stream contribution
        3. Patch into H2's Q, K, V inputs via W_Q, W_K, W_V
        4. Measure improvement in loss
        """

        # -----------------------------
        # Topology + data checks
        # -----------------------------
        if layer1 >= layer2:
            return 0.0

        if len(self.induction_pairs) == 0:
            return 0.0

        importances = []

        for pair in self.induction_pairs[:3]:
            clean_cache = pair["clean_cache"]
            corrupt_tokens = pair["corrupt_tokens"]
            corrupt_loss = pair["corrupt_loss"]

            try:
                # -----------------------------------------
                # 1. Get H1.z on BOTH clean AND corrupt runs
                # -----------------------------------------
                with torch.no_grad():
                    _, corrupt_cache = self.model.run_with_cache(
                        corrupt_tokens,
                        names_filter=lambda x: x == get_act_name("z", layer1)
                    )

                # Shapes: [batch, seq, n_heads, d_head]
                z_clean_full = clean_cache[get_act_name("z", layer1)]
                z_corrupt_full = corrupt_cache[get_act_name("z", layer1)]
                
                # Extract specific head: [batch, seq, d_head]
                z_clean = z_clean_full[:, :, head1, :]
                z_corrupt = z_corrupt_full[:, :, head1, :]

                # Δ signal sent by H1: [batch, seq, d_head]
                z_diff = z_clean - z_corrupt

                # -----------------------------------------
                # 2. Project H1.z_diff → residual stream
                # -----------------------------------------
                # W_O: [n_heads, d_head, d_model]
                W_O = self.model.W_O[layer1, head1]  # [d_head, d_model]

                # Project to residual stream: [batch, seq, d_model]
                # Using einsum for clarity and batch support
                resid_diff = torch.einsum("bsh,hd->bsd", z_diff, W_O)

                # -----------------------------------------
                # 3. Define path patch hook (Q, K, V)
                # -----------------------------------------
                def patch_qkv_input(activations, hook):
                    """
                    Patches Q, K, or V input of H2 with the signal from H1.
                    
                    Args:
                        activations: [batch, seq, n_heads, d_head]
                        hook: TransformerLens hook object
                    
                    Returns:
                        Modified activations with H1's signal added to H2
                    """
                    # Clone to avoid in-place modification
                    activations = activations.clone()
                    
                    # Select the appropriate weight matrix based on hook name
                    if "hook_q" in hook.name:
                        W = self.model.W_Q[layer2, head2]  # [d_model, d_head]
                    elif "hook_k" in hook.name:
                        W = self.model.W_K[layer2, head2]  # [d_model, d_head]
                    elif "hook_v" in hook.name:
                        W = self.model.W_V[layer2, head2]  # [d_model, d_head]
                    else:
                        return activations
                    
                    # Project residual stream contribution → specific input
                    # [batch, seq, d_model] @ [d_model, d_head] → [batch, seq, d_head]
                    input_patch = torch.einsum("bsd,dh->bsh", resid_diff, W)
                    
                    # Add the patch ONLY to the target head (head2)
                    activations[:, :, head2, :] += input_patch
                    
                    return activations

                # -----------------------------------------
                # 4. Run corrupt forward pass with edge patched
                # -----------------------------------------
                # Patch all three inputs (Q, K, V) to capture full edge effect
                patched_loss = self.model.run_with_hooks(
                    corrupt_tokens,
                    return_type="loss",
                    fwd_hooks=[
                        (f"blocks.{layer2}.attn.hook_q", patch_qkv_input),
                        (f"blocks.{layer2}.attn.hook_k", patch_qkv_input),
                        (f"blocks.{layer2}.attn.hook_v", patch_qkv_input)
                    ]
                ).item()

                # -----------------------------------------
                # 5. Edge importance = improvement in loss
                # -----------------------------------------
                # Positive value = edge restores performance
                # (moving corrupt loss closer to clean loss)
                improvement = corrupt_loss - patched_loss
                importances.append(improvement)

            except Exception as e:
                print(
                    f"  Warning: Edge patch failed "
                    f"L{layer1}.H{head1} → L{layer2}.H{head2}: {e}"
                )
                continue

        # Return mean improvement across all test pairs
        return float(np.mean(importances)) if importances else 0.0

    def format_graph_for_llm(self, target_layer: int, target_head: int,
                             top_k: int = 15) -> str:
        """Format attribution graph as text for LLM consumption"""
        if self.edge_matrix is None:
            self.build_attribution_graph()

        n_heads = self.model.cfg.n_heads
        target_idx = target_layer * n_heads + target_head

        edges = []
        for i in range(len(self.edge_matrix)):
            for j in range(len(self.edge_matrix)):
                if self.edge_matrix[i, j] > 0:
                    if i == target_idx or j == target_idx:
                        layer_i, head_i = divmod(i, n_heads)
                        layer_j, head_j = divmod(j, n_heads)
                        edges.append((self.edge_matrix[i, j], layer_i, head_i, layer_j, head_j))

        edges = sorted(edges, reverse=True)[:top_k]

        if not edges:
            return "No significant connections found in attribution graph."

        graph_text = f"Attribution Graph (connections involving L{target_layer}.H{target_head}):\n"
        for weight, li, hi, lj, hj in edges:
            direction = "→" if li < lj else "←"
            graph_text += f"  L{li}.H{hi} {direction} L{lj}.H{hj}  [importance: {weight:.3f}]\n"

        return graph_text


    # ============================================================
    # PART 2: CIRCUIT-LEVEL HYPOTHESIS GENERATION (FIXED)
    # ============================================================

    def generate_circuit_hypotheses(self, layer: int, head: int,
                                    max_hypotheses: int = 3) -> List[CircuitHypothesis]:
        """Generate circuit-level hypotheses using LLM"""
        if self.edge_matrix is None:
            self.build_attribution_graph()

        row = self.df[(self.df['layer'] == layer) & (self.df['head'] == head)].iloc[0]

        context = f"""Analyze this attention head and infer the complete circuit it participates in.

TARGET HEAD: Layer {layer}, Head {head}

OBSERVED METRICS:
- Induction score: {row['induction_score']:.3f} (high = copies previous tokens)
- Name-mover score: {row['name_mover_score']:.3f} (high = tracks entities)
- Positional score: {row['positional_score']:.3f} (high = fixed position bias)
- Attention entropy: {row['induction_entropy']:.3f} (low = focused attention)

{self.format_graph_for_llm(layer, head)}

Generate {max_hypotheses} CIRCUIT-LEVEL hypotheses. For each, specify:
1. circuit_path: List of [layer, head] pairs forming the pathway (e.g. [[0, 7], [5, 1]])
2. mechanism: High-level algorithm description
3. predicted_behavior: Testable prediction
4. test_strategy: Where should we look for the activation? 
   - "last" (for induction, movement, next-token prediction)
   - "max" (for detecting a specific word/feature anywhere in text)
   - "first" (for BOS or position 0 heads)

Return JSON array:
{{
  "hypotheses": [
    {{
      "circuit_path": [[0, 1]],
      "mechanism": "...",
      "predicted_behavior": "...",
      "test_strategy": "last"  <-- ADD THIS FIELD
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

            if "error" in response_data:
                raise ValueError(response_data["error"])

            hypotheses_data = response_data.get("hypotheses", [])

            hypotheses = []
            for h in hypotheses_data[:max_hypotheses]:
                hypotheses.append(CircuitHypothesis(
                    target_layer=layer,
                    target_head=head,
                    circuit_path=[(int(l), int(h)) for l, h in h.get('circuit_path', [[layer, head]])],
                    mechanism=h.get('mechanism', 'Unknown mechanism'),
                    predicted_behavior=h.get('predicted_behavior', 'Unknown behavior')
                ))

            self.hypotheses.extend(hypotheses)
            print(f"  ✓ Generated {len(hypotheses)} circuit hypotheses")
            return hypotheses

        except Exception as e:
            print(f"  Hypothesis generation failed: {e}")
            return [CircuitHypothesis(
                target_layer=layer,
                target_head=head,
                circuit_path=[(layer, head)],
                mechanism="Unknown - generation failed",
                predicted_behavior="Affects output distribution"
            )]


    # ============================================================
    # PART 3: AUTOMATED PROBE DESIGN (FIXED)
    # ============================================================

    def design_and_test_probe(self, hypothesis: CircuitHypothesis,
                              n_samples: int = 30) -> float:
        """Automated probe design and validation"""
        print(f"  Designing probe for: {hypothesis.mechanism[:50]}...")

        probe_prompt = f"""Generate {n_samples} labeled text examples to test: "{hypothesis.mechanism}"

Predicted behavior: {hypothesis.predicted_behavior}

Create {n_samples//2} POSITIVE examples (feature present) and {n_samples//2} NEGATIVE examples (feature absent).

Return JSON:
{{
  "examples": [
    {{"text": "The cat saw the cat", "has_feature": true}},
    {{"text": "Dogs like bones", "has_feature": false}}
  ]
}}
"""

        try:
            response_content = self.client.chat(
                model=MODEL_MAP["probe"],
                messages=[{"role": "user", "content": probe_prompt}],
                temperature=0.8,
                max_tokens=1500
            )

            response_data = json.loads(response_content)

            if "error" in response_data:
                raise ValueError(response_data["error"])

            dataset = response_data.get("examples", [])

        except Exception as e:
            print(f"    Probe dataset generation failed: {e}")
            return 0.0

        X_train = []
        y_train = []

        for item in dataset[:n_samples]:
            try:
                tokens = self.model.to_tokens(item['text'])
                _, cache = self.model.run_with_cache(tokens)
                
                # Get raw activation [seq_len, d_head]
                raw_z = safe_get_z(cache, hypothesis.target_layer, hypothesis.target_head)
                
                # --- DYNAMIC POOLING STRATEGY ---
                strategy = getattr(hypothesis, 'test_strategy', 'last') # Default to 'last'

                if strategy == 'last':
                    # Best for Induction, IOI, Previous Token
                    # "What is the head telling the NEXT token to do?"
                    activation = raw_z[-1, :].cpu().numpy()

                elif strategy == 'max':
                    # Best for "Find the needle in the haystack"
                    # "Did this feature appear ANYWHERE?"
                    # We take max absolute value or max norm to detect "firing"
                    # Simple approach: Max value across sequence for each dimension
                    activation = raw_z.max(dim=0).values.cpu().numpy()

                elif strategy == 'first':
                    # Best for BOS (Beginning of Sequence) heads
                    activation = raw_z[0, :].cpu().numpy()
                    
                elif strategy == 'mean':
                    # Fallback for "Mood/Sentiment" (Global properties)
                    activation = raw_z.mean(dim=0).cpu().numpy()
                
                else:
                    # Fallback
                    activation = raw_z[-1, :].cpu().numpy()

                X_train.append(activation)
                y_train.append(1 if item['has_feature'] else 0)

            except Exception as e:
                continue

        if len(X_train) < 10:
            print(f"    Insufficient data for probe training")
            return 0.0

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        try:
            probe = LogisticRegression(max_iter=500, random_state=42)
            scores = cross_val_score(probe, X_train, y_train, cv=3)
            probe_accuracy = scores.mean()

            print(f"    ✓ Probe accuracy: {probe_accuracy:.3f}")
            return float(probe_accuracy)

        except Exception as e:
            print(f"    Probe training failed: {e}")
            return 0.0


    # ============================================================
    # PART 4: OOD & ADVERSARIAL TEST GENERATION (FIXED)
    # ============================================================

    def generate_adversarial_prompts(self, hypothesis: CircuitHypothesis,
                                     n_prompts: int = 5) -> List[str]:
        """Generate out-of-distribution and adversarial test cases"""
        prompt_request = f"""Generate {n_prompts} ADVERSARIAL test prompts for: "{hypothesis.mechanism}"

Predicted behavior: {hypothesis.predicted_behavior}

Include:
1. Edge cases that break the hypothesis if wrong
2. Out-of-distribution examples (unusual tokens, syntax)
3. Counterfactuals testing boundary conditions

Examples:
- Pure repetition: "AAA AAA AAA"
- Nonsense: "xQz xQz mNp"
- Special chars: "🐱→🐱 🐱"
- Syntactic edge: "The the the the"

Return JSON:
{{
  "prompts": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]
}}
"""

        try:
            response_content = self.client.chat(
                model=MODEL_MAP["adversarial"],
                messages=[{"role": "user", "content": prompt_request}],
                temperature=0.9,
                max_tokens=500
            )

            response_data = json.loads(response_content)

            if "error" in response_data:
                raise ValueError(response_data["error"])

            prompts = response_data.get("prompts", [])
            print(f"  ✓ Generated {len(prompts)} adversarial test cases")
            return prompts[:n_prompts]

        except Exception as e:
            print(f"   Adversarial prompt generation failed: {e}")
            return [
                "AAA AAA AAA",
                "xQz xQz xQz",
                "The the the the the",
                "A B A B A B A B",
                "🐱 🐱 🐱"
            ]


    # ============================================================
    # PART 5: AUTOMATED INTERVENTION EXPERIMENTS (FIXED)
    # ============================================================

    def generate_intervention_experiment(self, hypothesis: CircuitHypothesis) -> Dict:
        """LLM designs custom intervention experiment"""
        prompt = f"""Design an intervention experiment to test: "{hypothesis.mechanism}"

Circuit path: {' → '.join([f'L{l}.H{h}' for l, h in hypothesis.circuit_path])}

Specify a causal intervention.

Return JSON:
{{
  "intervention_type": "ablate",
  "target_component": {{"layer": {hypothesis.target_layer}, "head": {hypothesis.target_head}}},
  "target_position": "all",
  "expected_outcome": "what should happen to model output"
}}
"""

        try:
            response_content = self.client.chat(
                model=MODEL_MAP["intervention"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=300
            )

            experiment = json.loads(response_content)

            if "error" in experiment:
                raise ValueError(experiment["error"])

            return experiment

        except Exception as e:
            print(f"   Intervention design failed: {e}")
            return {
                "intervention_type": "ablate",
                "target_component": {"layer": hypothesis.target_layer, "head": hypothesis.target_head},
                "target_position": "all",
                "expected_outcome": "Loss increases"
            }


    def execute_intervention(self, experiment: Dict, test_prompts: List[str]) -> Dict:
        """Execute LLM-designed intervention experiment"""
        intervention_type = experiment['intervention_type']
        target_layer = experiment['target_component']['layer']
        target_head = experiment['target_component']['head']

        results = []

        for prompt in test_prompts[:3]:
            tokens = self.model.to_tokens(prompt)
            baseline_loss = self.model(tokens, return_type="loss").item()

            if intervention_type == "ablate":
                def hook_ablate(z, hook):
                    z = z.clone()
                    z[:, :, target_head, :] = 0.0
                    return z

                modified_loss = self.model.run_with_hooks(
                    tokens, return_type="loss",
                    fwd_hooks=[(get_act_name("z", target_layer), hook_ablate)]
                ).item()

            elif intervention_type == "amplify":
                def hook_amplify(z, hook):
                    z = z.clone()
                    z[:, :, target_head, :] *= 2.0
                    return z

                modified_loss = self.model.run_with_hooks(
                    tokens, return_type="loss",
                    fwd_hooks=[(get_act_name("z", target_layer), hook_amplify)]
                ).item()

            else:
                modified_loss = baseline_loss

            results.append({
                "prompt": prompt,
                "baseline_loss": baseline_loss,
                "modified_loss": modified_loss,
                "delta": modified_loss - baseline_loss
            })

        return {
            "intervention": intervention_type,
            "average_effect": np.mean([r['delta'] for r in results]),
            "details": results
        }


    # ============================================================
    # PART 6: COMPREHENSIVE VALIDATION
    # ============================================================

    def validate_circuit_hypothesis(self, hypothesis: CircuitHypothesis,
                                    human_review: bool = True) -> Dict:
        """Full validation pipeline combining all metrics"""
        print(f"\n{'='*70}")
        print(f"🔬 VALIDATING CIRCUIT HYPOTHESIS")
        print(f"{'='*70}")
        print(f"Mechanism: {hypothesis.mechanism}")
        print(f"Circuit: {' → '.join([f'L{l}.H{h}' for l, h in hypothesis.circuit_path])}")

        # Metric 1: Automated probe design
        probe_score = self.design_and_test_probe(hypothesis)
        hypothesis.validation_scores['probe_accuracy'] = probe_score

        # Metric 2: Adversarial testing
        adv_prompts = self.generate_adversarial_prompts(hypothesis)
        kl_scores = []
        for prompt in adv_prompts:
            tokens = self.model.to_tokens(prompt)
            baseline_logits = self.model(tokens, return_type="logits")[0, -1, :]
            baseline_probs = torch.softmax(baseline_logits, dim=-1)

            def hook_z(z, hook):
                z = z.clone()
                z[:, :, hypothesis.target_head, :] = 0.0
                return z

            try:
                patched_logits = self.model.run_with_hooks(
                    tokens, return_type="logits",
                    fwd_hooks=[(get_act_name("z", hypothesis.target_layer), hook_z)]
                )[0, -1, :]
                patched_probs = torch.softmax(patched_logits, dim=-1)

                kl = torch.nn.functional.kl_div(
                    patched_probs.log(), baseline_probs, reduction='sum'
                ).item()
                kl_scores.append(kl)
            except:
                continue

        adv_kl_score = float(np.mean(kl_scores)) if kl_scores else 0.0
        hypothesis.validation_scores['adversarial_kl'] = adv_kl_score

        # Metric 3: Custom intervention experiments
        intervention_exp = self.generate_intervention_experiment(hypothesis)
        intervention_result = self.execute_intervention(intervention_exp, adv_prompts)
        intervention_score = abs(intervention_result['average_effect'])
        hypothesis.validation_scores['intervention_effect'] = intervention_score

        # Metric 4: Faithfulness
        faith_scores = []
        if len(self.induction_pairs) > 0:
            for pair in self.induction_pairs:
                delta = patch_head_delta(pair, hypothesis.target_layer, hypothesis.target_head, self.model)
                faith_scores.append(abs(delta) / (pair['corrupt_loss'] + 1e-8))

        faith_score = float(np.mean(faith_scores)) if faith_scores else 0.0
        hypothesis.validation_scores['faithfulness'] = faith_score

        # Combined validation score
        overall_score = (
            probe_score * 0.35 +
            adv_kl_score * 0.25 +
            intervention_score * 0.25 +
            faith_score * 0.15
        )

        hypothesis.confidence = overall_score

        result = {
            'hypothesis': hypothesis.mechanism,
            'circuit_path': hypothesis.circuit_path,
            'probe_accuracy': probe_score,
            'adversarial_kl': adv_kl_score,
            'intervention_effect': intervention_score,
            'faithfulness': faith_score,
            'overall_score': overall_score,
            'verdict': 'SUPPORTED' if overall_score > 0.5 else 'WEAK' if overall_score > 0.3 else 'REJECTED'
        }

        # Human review for borderline cases
        if human_review and 0.3 < overall_score < 0.7:
            print(f"\n{'='*70}")
            print(f" HUMAN REVIEW REQUESTED")
            print(f"{'='*70}")
            print(f"Circuit: {' → '.join([f'L{l}.H{h}' for l, h in hypothesis.circuit_path])}")
            print(f"Mechanism: {hypothesis.mechanism}")
            print(f"\nValidation Scores:")
            print(f"  - Probe Accuracy: {probe_score:.3f}")
            print(f"  - Adversarial KL: {adv_kl_score:.3f}")
            print(f"  - Intervention Effect: {intervention_score:.3f}")
            print(f"  - Faithfulness: {faith_score:.3f}")
            print(f"  - Overall: {overall_score:.3f}")

            user_input = input("\n Accept this hypothesis? (y/n/skip): ").lower()
            if user_input == 'n':
                result['verdict'] = 'HUMAN_REJECTED'
            elif user_input == 'y':
                result['verdict'] = 'HUMAN_CONFIRMED'
                hypothesis.confidence = 1.0

        print(f"\n{'='*70}")
        print(f"VERDICT: {result['verdict']} (score: {overall_score:.3f})")
        print(f"{'='*70}")

        return result


    # ============================================================
    # PART 7: MAIN AUTOMATION LOOP
    # ============================================================

    def run_full_automation_loop(self, top_k_heads: int = 5,
                                 max_iterations: int = 2):
        """Complete automated interpretability pipeline"""
        print("\n" + "="*70)
        print(" FULL AUTOMATED INTERPRETABILITY PIPELINE")
        print("="*70)
        print(f"Analyzing top {top_k_heads} heads")
        print(f"Max {max_iterations} iterations per head")
        print(f"Using GWDG LLM API: https://chat-ai.academiccloud.de/v1")
        print("="*70)

        self.build_attribution_graph(top_k_heads=top_k_heads * 2)

        top_heads = self.df.nlargest(top_k_heads, 'induction_score')[['layer', 'head']]

        all_results = []

        for idx, row in top_heads.iterrows():
            layer, head = int(row['layer']), int(row['head'])

            print(f"\n{'='*70}")
            print(f" ANALYZING HEAD: Layer {layer}, Head {head}")
            print(f"{'='*70}")

            head_results = []

            for iteration in range(max_iterations):
                print(f"\n Iteration {iteration + 1}/{max_iterations}")

                hypotheses = self.generate_circuit_hypotheses(layer, head, max_hypotheses=2)

                if not hypotheses:
                    print("   No hypotheses generated")
                    break

                for hyp in hypotheses:
                    result = self.validate_circuit_hypothesis(
                        hyp,
                        human_review=(iteration == 0)
                    )
                    head_results.append(result)

                    if result['overall_score'] > 0.7:
                        print(f"\n Strong hypothesis found! Moving to next head.")
                        break

                if head_results and head_results[-1]['overall_score'] > 0.7:
                    break

            all_results.append({
                'layer': layer,
                'head': head,
                'hypotheses': head_results
            })

        return all_results


    def generate_comprehensive_report(self, results: List[Dict]) -> str:
        """Generate detailed report with all findings"""
        report = f"""#  Automated Circuit Discovery Report

## Methodology

This analysis uses:
1. **Attribution Graph Extraction** - ACDC-style edge importance
2. **Circuit-Level Reasoning** - LLM interprets computational graphs
3. **Automated Probe Design** - Validates predicted features
4. **Adversarial Testing** - OOD and counterfactual inputs
5. **Custom Interventions** - LLM-designed causal experiments

## Summary Statistics

- Heads Analyzed: {len(results)}
- Total Hypotheses: {sum(len(r['hypotheses']) for r in results)}
- Supported (>0.5): {sum(1 for r in results for h in r['hypotheses'] if h['overall_score'] > 0.5)}
- Weak (0.3-0.5): {sum(1 for r in results for h in r['hypotheses'] if 0.3 <= h['overall_score'] <= 0.5)}
- Rejected (<0.3): {sum(1 for r in results for h in r['hypotheses'] if h['overall_score'] < 0.3)}

## Discovered Circuits

"""

        for r in results:
            report += f"\n### Layer {r['layer']}, Head {r['head']}\n\n"

            for i, hyp in enumerate(r['hypotheses'], 1):
                report += f"**Hypothesis {i}**: {hyp['hypothesis']}\n\n"
                report += f"**Circuit Path**: {' → '.join([f'L{l}.H{h}' for l, h in hyp['circuit_path']])}\n\n"
                report += f"**Verdict**: {hyp['verdict']} (Overall Score: {hyp['overall_score']:.3f})\n\n"
                report += f"**Validation Metrics**:\n"
                report += f"- Probe Accuracy: {hyp['probe_accuracy']:.3f}\n"
                report += f"- Adversarial KL Divergence: {hyp['adversarial_kl']:.3f}\n"
                report += f"- Intervention Effect: {hyp['intervention_effect']:.3f}\n"
                report += f"- Faithfulness: {hyp['faithfulness']:.3f}\n\n"
                report += "---\n\n"

        report += """
## Interpretation Notes

### High Probe Accuracy (>0.7)
Indicates the head reliably represents the predicted feature in its activations.

### High Adversarial KL (>0.5)
The head significantly affects output distribution even on OOD inputs - strong causal role.

### High Intervention Effect (>0.3)
Custom interventions produce measurable effects - confirms causal importance.

### High Faithfulness (>0.5)
Activation patching shows the head is causally necessary for the behavior.

## Next Steps

1. Investigate top-scoring circuits in detail
2. Test on additional task-specific datasets
3. Extend to deeper circuit analysis (3+ component interactions)
4. Compare findings with manual interpretability research

---
*Generated by Advanced Automated Interpretability Agent*
*Methodology based on ACDC [Conmy et al.], MAIA [Rott Shaham et al.], and Neel Nanda's research*
"""

        return report



