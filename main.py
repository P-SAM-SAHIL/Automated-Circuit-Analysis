


# ============================================================
# MAIN ENTRY
# ============================================================
import json
import numpy as np
import pandas as pd
import torch
import argparse

from transformer_lens import HookedTransformer
from llm_client import RobustLLMClient
from config import GWDG_BASE_URL, MODEL_MAP

from core.interpreter import AdvancedAutomatedInterpreter
from core.task_generator import TaskGenerator
from utils.patching import patch_head_delta


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_head_structural_metrics(model, tokens, layer, head, attention_pattern):
    """
    Computes structural metrics (Induction, Copying, etc.) for a specific head.
    This prevents the KeyError in the Interpreter by providing the expected columns.
    """
    # Pattern: [batch, seq_len, seq_len]
    attn = attention_pattern[:, head, :, :]
    batch_size, seq_len, _ = attn.shape
    
    # 1. INDUCTION ENTROPY
    entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)
    avg_entropy = entropy.mean().item()

    # 2. POSITIONAL SCORE (Attention to previous token, diag -1)
    score_prev = 0.0
    for b in range(batch_size):
        diag = torch.diagonal(attn[b], offset=-1)
        score_prev += diag.mean().item()
    avg_positional = score_prev / batch_size

    # 3. INDUCTION & COPYING SCORES
    induction_score = 0.0
    copying_score = 0.0

    for b in range(batch_size):
        seq_ind_score = []
        seq_copy_score = []
        current_tokens = tokens[b]
        
        for i in range(1, seq_len):
            # Find previous occurrences of the current token
            prev_instances = (current_tokens[:i] == current_tokens[i])
            
            if prev_instances.any():
                # Indices where this token appeared before
                prev_idxs = torch.nonzero(prev_instances).squeeze(-1)
                
                # COPYING: Attention to the previous instance itself
                seq_copy_score.append(attn[b, i, prev_idxs].sum().item())
                
                # INDUCTION: Attention to the token *after* the previous instance
                ind_idxs = prev_idxs + 1
                ind_idxs = ind_idxs[ind_idxs < i] # Ensure we don't look ahead
                if len(ind_idxs) > 0:
                    seq_ind_score.append(attn[b, i, ind_idxs].sum().item())

        if seq_ind_score: induction_score += np.mean(seq_ind_score)
        if seq_copy_score: copying_score += np.mean(seq_copy_score)

    return {
        "induction_entropy": avg_entropy,
        "positional_score": avg_positional,
        "induction_score": induction_score / batch_size,
        "name_mover_score": copying_score / batch_size
    }


def build_task_head_metrics_df(model, pair_data, max_pairs=3):
    print("Computing structural and task metrics per head...")
    
    # Get tokens from the first pair to analyze structure
    if "tokens" in pair_data[0]["clean_cache"]:
        example_tokens = pair_data[0]["clean_cache"]["tokens"]
    else:
        # Fallback if tokens aren't explicitly cached
        example_tokens = model.to_tokens(pair_data[0]["clean_text"])

    # Run ONE pass to get all attention patterns efficiently
    _, cache = model.run_with_cache(
        example_tokens, 
        names_filter=lambda x: x.endswith("pattern")
    )
    
    rows = []
    for layer in range(model.cfg.n_layers):
        # Get pattern for whole layer [batch, n_heads, seq, seq]
        layer_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        
        for head in range(model.cfg.n_heads):
            # 1. Compute Patching Score (Task Importance)
            deltas = []
            for pair in pair_data[:max_pairs]:
                delta = patch_head_delta(pair, layer, head, model)
                deltas.append(abs(delta))
            
            # 2. Compute Structural Scores (Induction, etc.)
            struct_metrics = calculate_head_structural_metrics(
                model, example_tokens, layer, head, layer_pattern
            )

            row = {
                "layer": layer,
                "head": head,
                "task_score": float(np.mean(deltas)),
                # Merge the new metrics
                **struct_metrics
            }
            rows.append(row)
            
    return pd.DataFrame(rows)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced Automated Interpreter for Circuit Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --model gpt2-medium --topkheads 5
  python main.py --apikey YOUR_KEY --npairs 10
  python main.py --targetbehaviors "Induction" "Previous Token Head"
  python main.py --device cpu --maxiterations 3
        """
    )
    
    parser.add_argument("--model", default="gpt2-small", 
                       help="TransformerLens model name (default: gpt2-small)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device: cuda or cpu (default: auto-detect)")
    parser.add_argument("--topkheads", type=int, default=3, 
                       help="Top K heads to analyze (default: 3)")
    parser.add_argument("--maxiterations", type=int, default=2, 
                       help="Max iterations per head (default: 2)")
    parser.add_argument("--npairs", type=int, default=5, 
                       help="Number of ACDC pairs per task (default: 5)")
    parser.add_argument("--maxpairs", type=int, default=3, 
                       help="Max pairs for metric computation (default: 3)")
    parser.add_argument("--apikey", default="", 
                       help="LLM API key (default: empty/load from env)")
    parser.add_argument("--baseurl", default=GWDG_BASE_URL, 
                       help=f"LLM base URL (default: {GWDG_BASE_URL})")
    parser.add_argument("--targetbehaviors", nargs="*", 
                       default=[
                           "Induction (copying previous token)",
                           "Indirect Object Identification (John gave Mary a drink)",
                           "Previous Token Head (attending to immediate neighbor)"
                       ], 
                       help="List of behaviors to investigate")
    parser.add_argument("--output", default="multi_task_discovery_report.md", 
                       help="Output report filename (default: multi_task_discovery_report.md)")
    
    args = parser.parse_args()
    
    # ------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------
    print("\n" + "="*70)
    print(" INITIALIZING ADVANCED AUTOMATED INTERPRETER")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Top K Heads: {args.topkheads}")
    print(f"Max Iterations: {args.maxiterations}")
    print(f"ACDC Pairs: {args.npairs}")
    print(f"Target Behaviors: {len(args.targetbehaviors)}")
    print("="*70 + "\n")
    
    client = RobustLLMClient(
        base_url=args.baseurl,
        api_key=args.apikey
    )
    
    model = HookedTransformer.from_pretrained(
        args.model,
        device=args.device
    )
    model.eval()
    
    task_gen = TaskGenerator(client, model)
    
    # ------------------------------------------------------------
    # Main Analysis Loop
    # ------------------------------------------------------------
    full_report = ""
    
    for behavior in args.targetbehaviors:
        print("\n" + "="*80)
        print(f"🚀 STARTING INVESTIGATION: {behavior}")
        print("="*80)
        
        dynamic_pair_data = task_gen.generate_acdc_data(
            behavior,
            n_pairs=args.npairs
        )
        
        if not dynamic_pair_data:
            print("Skipping due to data generation failure.")
            continue
            
        df_task_metrics = build_task_head_metrics_df(
            model=model,
            pair_data=dynamic_pair_data,
            max_pairs=args.maxpairs
        )
        
        advanced_interp = AdvancedAutomatedInterpreter(
            model=model,
            df_results=df_task_metrics,
            pair_data_induction=dynamic_pair_data,
            pair_data_name_mover=[],
            client=client
        )
        
        results = advanced_interp.run_full_automation_loop(
            top_k_heads=args.topkheads,
            max_iterations=args.maxiterations
        )
        
        report_section = advanced_interp.generate_comprehensive_report(results)
        full_report += f"\n\n# RESULTS FOR: {behavior}\n" + report_section
    
    # ------------------------------------------------------------
    # Save Report
    # ------------------------------------------------------------
    with open(args.output, "w") as f:
        f.write(full_report)
    
    print("\n" + "="*70)
    print(f"✓ Full multi-task exploration complete!")
    print(f"✓ Report saved to: {args.output}")
    print("="*70 + "\n")
