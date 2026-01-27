import argparse
import torch
import sys  
from transformer_lens import HookedTransformer
from llm_client import RobustLLMClient
from config import GWDG_BASE_URL
from core.task_generator import TaskGenerator
from core.interpreter import AdvancedAutomatedInterpreter

# --- NEW LOGGER CLASS ---
class Logger(object):
    """Duplicates output to both the terminal and a log file."""
    def __init__(self, filename="experiment_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8") # 'w' overwrites, 'a' appends

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Needed for compatibility with python's flush command
        self.terminal.flush()
        self.log.flush()

def main():
    parser = argparse.ArgumentParser(description="Automated Mechanistic Interpretability")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--apikey", required=True, help="Your LLM API Key")
    parser.add_argument("--npairs", type=int, default=5, help="dataset size")
    parser.add_argument("--threshold", type=float, default=0.05, help="ACDC pruning threshold")
    parser.add_argument("--behaviors", nargs="+", default=["Induction", "indirect Object Identification"])
    
    # <--- NEW ARGUMENT FOR FILE NAME
    parser.add_argument("--output_file", default="final_report.txt", help="Name of the output log file")
    
    args = parser.parse_args()

    # <--- START LOGGING
    # This redirects all 'print' statements to the Logger class
    sys.stdout = Logger(args.output_file)
    print(f"📄 Logging started. Output will be saved to: {args.output_file}")

    # --- INITIALIZE ---
    client = RobustLLMClient(base_url=GWDG_BASE_URL, api_key=args.apikey)
    model = HookedTransformer.from_pretrained(
        args.model, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # 1. Fixes "Need to be able to see hook MLP inputs"
        use_hook_mlp_in=True,
        
        # 2. Prevent future errors (ACDC needs these too)
        use_attn_result=True,
        use_split_qkv_input=True,
        
  
    )
    
    task_gen = TaskGenerator(client, model)
    interpreter = AdvancedAutomatedInterpreter(model, client)

    full_results = []

    for behavior in args.behaviors:
        print(f"\n{'='*80}\n🚀 ANALYZING: {behavior}\n{'='*80}")

        # STEP 1: Task-Specific Data Generation
        data = task_gen.generate_acdc_data(behavior, n_pairs=args.npairs)
        if not data: continue

        # STEP 2: Causal Attribution Graph Construction (ACDC)
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
        
        # Optional: Add detailed breakdown to the text file
        print(f"      Probe Accuracy: {res['probe_accuracy']:.2f}")
        print(f"      Intervention Impact: {res['intervention_effect']:.2%}")
        print("-" * 40)

if __name__ == "__main__":
    main()