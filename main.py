# main.py
import argparse
import torch
from transformer_lens import HookedTransformer
from llm_client import RobustLLMClient
from config import GWDG_BASE_URL
from core.task_generator import TaskGenerator
from core.interpreter import AdvancedAutomatedInterpreter

def main():
    parser = argparse.ArgumentParser(description="Automated Mechanistic Interpretability")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--apikey", required=True, help="Your LLM API Key")
    parser.add_argument("--npairs", type=int, default=5, help="dataset size")
    parser.add_argument("--threshold", type=float, default=0.05, help="ACDC pruning threshold")
    parser.add_argument("--behaviors", nargs="+", default=["Induction", "indirect Object Identification"])
    
    args = parser.parse_args()

    # --- INITIALIZE ---
    client = RobustLLMClient(base_url=GWDG_BASE_URL, api_key=args.apikey)
    model = HookedTransformer.from_pretrained(
        args.model, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Pass the config flags here instead
        fold_ln=False, 
        center_writing_weights=False, 
        center_unembed=False,
        # If you wanted these features:
        # fold_value_biases=False,
    )
    
   
    model.set_use_split_qkv_input(True)
    
    task_gen = TaskGenerator(client, model)
    interpreter = AdvancedAutomatedInterpreter(model, client)

    full_results = []

    for behavior in args.behaviors:
        print(f"\n{'='*80}\n🚀 ANALYZING: {behavior}\n{'='*80}")

        # STEP 1: Task-Specific Data Generation
        data = task_gen.generate_acdc_data(behavior, n_pairs=args.npairs)
        if not data: continue

        # STEP 2: Causal Attribution Graph Construction (ACDC)
        # This populates interpreter.edge_matrix and interpreter.current_graph_edges
        interpreter.build_attribution_graph(data, threshold=args.threshold)

        # STEP 3: Graph -> High-Level Causal Interpretation (LLM)
        hypotheses = interpreter.generate_circuit_hypotheses(task_description=behavior)

        # STEP 4 & 5: Automated Validation & Verdict
        for hyp in hypotheses:
            verdict = interpreter.validate_circuit_hypothesis(hyp)
            full_results.append(verdict)

    # --- FINAL REPORT ---
    print("\n" + "="*80 + "\nFINAL SUMMARY REPORT\n" + "="*80)
    for res in full_results:
        print(f"[{res['verdict']}] Mechanism: {res['hypothesis'][:50]}...")
        print(f"      Score: {res['overall_score']:.2f} | Path: {res['circuit_path']}")

if __name__ == "__main__":
    main()

    """
    python main.py --model gpt2-small --apikey "your_api_key_here" --behaviors "Induction"
    """