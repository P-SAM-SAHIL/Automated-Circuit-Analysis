# Automated Circuit Analysis - Model Version 2

Massive changes have been introduced from Version 1.  
For full documentation, please refer to:  
[Model v2 Documentation](https://docs.google.com/document/d/1zLDJbHIbrZZW81yNFEMKEFA-lcA9ikAeHAkOgEDTXMQ/edit?usp=sharing)

---

##  Clone Repository
```bash
!git clone https://github.com/P-SAM-SAHIL/Automated-Circuit-Analysis.git
```
## Packages required
``` 
!pip install transformer_lens openai wandb scikit-learn einops jaxtyping
!apt install libgraphviz-dev
!pip install pygraphviz
```
## Navigate to Project Directory

```
%cd /content/Automated-Circuit-Analysis
```
## Example run
```
python main.py \
  --model "gpt2-small" \
  --apikey "" \
  --npairs 50 \
  --threshold 0.05 \
  --behaviors "Induction" \
  --output_file "experiment_results_v2.txt"
```
