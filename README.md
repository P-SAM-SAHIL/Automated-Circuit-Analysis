for full documentation follow this link : https://docs.google.com/document/d/1zLDJbHIbrZZW81yNFEMKEFA-lcA9ikAeHAkOgEDTXMQ/edit?usp=sharing

Clone : !git clone https://github.com/P-SAM-SAHIL/Automated-Circuit-Analysis.git \
Packages required :
!pip install transformer_lens openai wandb scikit-learn einops jaxtyping \
!apt install libgraphviz-dev \
!pip install pygraphviz  \

%cd /content/Automated-Circuit-Analysis

Example for Runing : \
python main.py \
  --model "gpt2-small" \
  --apikey "" \
  --npairs 50 \
  --threshold 0.05 \
  --behaviors "Induction" \
  --output_file "experiment_results_v2.txt"
