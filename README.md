# README

## Automating Hypothesis Generation and Validation in Mechanistic Interpretability

### 1. Research Question

This project asks:

**Can we automate the full hypothesis generation and validation loop in mechanistic interpretability using LLM agents?**

Specifically, we investigate:

* Can LLMs generate **mechanistic hypotheses** about how a transformer performs a task?
* Can these hypotheses be **validated automatically** using probes, interventions, and synthetic/OOD data?
* How dependent is LLM reasoning on the quality of the extracted **attribution graph** (ACDC-style)?

The goal is not just automation of experiments, but automation of the **scientific workflow** itself.

---

### 2. High-Level Algorithm

The system implements a **zero-touch mechanistic pipeline** inspired by ACDC and MAIA:

#### Phase 1: Data & Cache Generation

* Synthetic task data is generated (e.g., Induction, Previous Token).
* Clean and corrupted runs are cached using `model.run_with_cache()`.
* These caches form the baseline for all patching and interventions.

#### Phase 2: Metric Computation & Head Selection

* Every attention head is profiled using structural metrics:

  * Induction Score
  * Positional Score
  * Name Mover Score
  * Induction Entropy
* Heads are ranked (currently by `induction_score`), and the top-K heads are selected for analysis.

#### Phase 3: Attribution Graph Construction (ACDC-Style)

* Pairwise path patching is used to test causal influence between heads.
* If patching a sender head into a receiver improves loss, an edge is added:

  ```
  Lx.Hi → Ly.Hj
  ```
* The result is a sparse attribution graph (or empty if signal is weak).

#### Phase 4: LLM-Driven Hypothesis Generation

* The LLM receives:

  * Head metrics
  * Attribution graph edges
* It outputs a **mechanistic hypothesis** in structured JSON:

  * `circuit_path`
  * `mechanism`

#### Phase 5: Automated Validation (MAIA-Style)

Each hypothesis is tested using four automated experiments:

1. **Linear Probes** (feature presence)
2. **Adversarial Prompts** (robustness)
3. **Interventions** (causal effect)
4. **Faithfulness** (sufficiency via recovery)

All scores are aggregated into a final verdict.

---

### 3. Code Structure

```
.
├── main.py
│   ├── Entry point
│   ├── Argument parsing (task, dataset size)
│   ├── Orchestrates full pipeline
│
├── core/
│   ├── task_generator.py
│   │   └── Generates clean/corrupt task data
│   │
│   ├── interpreter.py
│   │   ├── Attribution graph construction
│   │   ├── Hypothesis generation (LLM)
│   │   └── Hypothesis validation loop
│
├── utils/
│   ├── patching.py
│   │   ├── Head patching
│   │   ├── Edge importance computation
│   │
│   ├── probing.py
│   │   └── Linear probe training & evaluation
│
├── reports/
│   └── Auto-generated markdown reports
│
└── README.md
```

---

### 4. Key Takeaway

The system **successfully automates the scientific process** (hypothesis generation, experiment design, validation), but struggles on **distributed circuits** due to:

* Single-head sufficiency assumptions
* Top-K head filtering
* Conservative ACDC thresholds

This highlights a core result: **automation works**, but the *microscope* must match the distributed nature of transformer computation.

---


