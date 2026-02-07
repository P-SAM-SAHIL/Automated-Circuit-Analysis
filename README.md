# Automated Circuit Analysis - Model Version 2

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
!pip install cmapy
```
## Navigate to Project Directory

```
%cd /content/Automated-Circuit-Analysis
%mkdir ims
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


---

# Automated Circuit Analysis

### End-to-End Hypothesis Generation and Validation for Mechanistic Interpretability

---

## Acknowledgements

**Special thanks to:**

* **Dr. Fazl Barez** (University of Oxford) — for introducing the core research direction of *automated circuit discovery and validation*.
* **Dr. Anne Lauscher** (University of Hamburg) — for research discussions and feedback shaping the problem formulation.

---

## Research Question

We ask:

> **Can the full hypothesis generation and validation loop in mechanistic interpretability be automated using LLM agents, without human tuning or manual inspection?**

This decomposes into the following sub-questions:

### Automated Hypothesis Generation

* Can LLMs infer **high-level causal graphs** for a task directly?
* Does allowing an agent to **actively select diverse prompts and observe model behavior** improve hypothesis quality?
* How well can LLMs **interpret attribution graphs**, and how much does structured prompting improve this?

### Automated Validation

* Can probes be **automatically designed** to test for predicted features?
* Can intervention experiments—including **synthetic and out-of-distribution (OOD) inputs**—be fully automated?
* Can hypotheses be **accepted or rejected algorithmically**, without human judgment?

---

## System Overview

This work presents a **fully automated, end-to-end pipeline** for discovering, interpreting, and validating **task-specific causal circuits** in transformer models.

The system integrates:

* **Causal attribution** (ACDC)
* **LLM-based mechanistic interpretation**
* **Fully mechanized validation**



---

## Architecture (Work in Progress)

### 1. Task-Specific Data Generation (LLM-Driven)

**Input**

* Target behavior (e.g., induction, indirect object identification, positional dependence)

**Process**

* An LLM generates *clean / corrupt* prompt pairs designed to isolate the target behavior.

**Clean vs. Corrupt Prompt Constraints**

* Identical token length
* Identical token positions
* Differ only in the **causal variable relevant to the task**

**Output**

* Clean / corrupt token pairs suitable for causal attribution


---

### 2. Causal Attribution Graph Construction (ACDC)

**Input**

* Clean tokens
* Corrupt tokens

**Method**

* Automated Circuit Discovery (ACDC) with recursive pruning

**Metric**

* KL divergence:
  [
  \text{KL}(\text{clean} \parallel \text{patched})
  ]
  measured at the prediction position

**Output**

* A task-specific causal attribution graph

**Graph Semantics**

* **Nodes:** attention heads, MLP blocks, input components
* **Edges:** causally necessary information flow identified via pruning

---

### 3. Attribution Graph → High-Level Causal Interpretation (LLM)

**Input**

* Attribution graph (edges with direction and effect size)
* Task description

**Process**

* A *single* LLM pass synthesizes the graph into a coherent mechanistic explanation

**Output**
One causal hypothesis per task, consisting of:

* Circuit path(s)
* Algorithmic mechanism description
* Falsifiable behavioral prediction

**Constraints**

* No iterative regeneration
* No head hallucination
* Interpretation must rely **only** on the graph and task description

---

### 4. Automated Validation (Fully Mechanized)

> No human tuning, thresholds, or manual inspection.

#### 4.1 Probe Generation

* LLM generates minimal *clean vs. broken* examples
* Linear probe trained on hypothesized head or MLP activations
* **Metric:** cross-validated probe accuracy

#### 4.2 Intervention Tests

* Zero-ablation of the hypothesized circuit
* Measure:

  * Change in loss
  * Change in KL divergence

#### 4.3 OOD / Adversarial Inputs

* LLM generates structure-preserving counterfactuals
* Tests causal **stability** of the circuit

---

### 5. Verdict Layer

**Inputs**

* Probe accuracy
* Intervention effect size
* Faithfulness (KL divergence between full model and circuit)

**Output**

* **Supported**
* **Weak**
* **Rejected**

> Verdicts are produced by a **fixed scoring rule**, not human judgment.

---

## What This Architecture Evaluates

* Whether task-specific data induces **task-specific causal graphs**
* Whether LLMs can infer **high-level causal structure** from attribution graphs
* Whether probes, interventions, and adversarial tests can be **fully automated**
* Whether LLM-generated mechanistic explanations are **causally supported**

---

## Experimental Results

### Induction Task — GPT-2 Small

**Model**

* `gpt2-small`

**Task**

* Induction

**Dataset**

* 4 clean / corrupt prompt pairs

**ACDC Configuration**

* Metric: KL(clean || patched) at final token
* Pruning threshold: `0.005`

---

### Attribution Graph

* ACDC completed with **207 active edges**
* Due to graph size, the **top 50 edges** (by effect size) were passed to the LLM
* Active edges concentrated in **upper layers (≈ layers 8–11)**
  → consistent with known induction behavior

---

### Circuit Hypothesis

* **Number of hypotheses:** 1

**Identified Circuit Path**

```
[(L0, MLP) → (L0, MLP) → (L3, H0) → (L4, H3) → (L4, H3)]
```

**High-Level Mechanism**

> Input is processed through early-layer MLPs, then routed through a small set of late-layer attention heads to support induction-style token copying.

---

### Automated Validation Results

* **Probe accuracy:** `0.93`
  (Linear probe on activations of `L4.H3`)
* **Intervention test:**
  Zero-ablation caused **+4.39% average loss** across 5 adversarial prompts
* **Faithfulness (KL divergence):** `4.78`
  → Circuit alone does not fully reconstruct model behavior

---

## Limitations & Open Questions

### Attribution Method

* ACDC relies on **activation patching**
* Faster and more scalable alternatives:

  * Attribution Patching (EAP)
  * AtP*

### Representation Questions

* SAE / dictionary learning vs. direct circuit graphs
* Monosemantic vs. polysemantic attribution
* Linear probes vs. **circuit-aware probes**

### Interpretation Quality

* Improving hypothesis clarity and abstraction
* Reducing reliance on top-K edge truncation

### Scalability

* Can this pipeline scale to **LLaMA-8B and beyond**?
* What classes of behavior can be reliably discovered?

---

## References

* *Circuit Tracing: Revealing Computational Graphs in Language Models*
* *Attribution Patching: Activation Patching at Industrial Scale* (2310.10348)
* *Attribution Patching Outperforms Automated Circuit Discovery*
* *Position-Aware Automatic Circuit Discovery* (2502.04577)
* *Automatic Discovery of Visual Circuits* (2404.14349)
* *Sparse Autoencoders Enable Scalable Circuit Identification*
* *Hypothesis Testing the Circuit Hypothesis in LLMs*
* *Uncovering Intermediate Variables in Transformers Using Circuit Probing* (2311.04354)
* *Automatically Identifying Local and Global Circuits* (2405.13868)
* *AtP*: Efficient Localization of LLM Behavior* (2403.00745)
* *Have Faith in Faithfulness* (2403.17806)
* *Language Model Circuits Are Sparse in the Neuron Basis* — Transluce AI
* Fazl Barez, *Automated Interpretability Research Agenda*

---


