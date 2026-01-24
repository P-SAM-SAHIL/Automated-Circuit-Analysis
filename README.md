for full documentation follow this link : https://docs.google.com/document/d/1zLDJbHIbrZZW81yNFEMKEFA-lcA9ikAeHAkOgEDTXMQ/edit?usp=sharing

Packages required : pip install transformer-lens circuitsvis networkx\

Example for Runing : \
python main.py \
  --model gpt2-small \
  --device cuda \
  --npairs 30 \
  --maxpairs 30 \
  --topkheads 20 \
  --targetbehaviors "Induction" \
   --output my_report.md \
  --apikey ""
