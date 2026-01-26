

# RESULTS FOR: Induction (copying previous token)
#  Automated Circuit Discovery Report

## Methodology

This analysis uses:
1. **Attribution Graph Extraction** - ACDC-style edge importance
2. **Circuit-Level Reasoning** - LLM interprets computational graphs
3. **Automated Probe Design** - Validates predicted features
4. **Adversarial Testing** - OOD and counterfactual inputs
5. **Custom Interventions** - LLM-designed causal experiments

## Summary Statistics

- Heads Analyzed: 3
- Total Hypotheses: 12
- Supported (>0.5): 0
- Weak (0.3-0.5): 2
- Rejected (<0.3): 10

## Discovered Circuits


### Layer 5, Head 5

**Hypothesis 1**: Directly attends to the current token based on its positional information

**Circuit Path**: L5.H5

**Verdict**: WEAK (Overall Score: 0.333)

**Validation Metrics**:
- Probe Accuracy: 0.933
- Adversarial KL Divergence: 0.017
- Intervention Effect: 0.009
- Faithfulness: 0.000

---

**Hypothesis 2**: Copies the token following a prefix via simple induction and positional encoding

**Circuit Path**: L4.H0 → L5.H5

**Verdict**: REJECTED (Overall Score: 0.207)

**Validation Metrics**:
- Probe Accuracy: 0.567
- Adversarial KL Divergence: 0.027
- Intervention Effect: 0.008
- Faithfulness: 0.000

---

**Hypothesis 3**: Directly attends to the current token based on its position and content

**Circuit Path**: L5.H5

**Verdict**: REJECTED (Overall Score: 0.207)

**Validation Metrics**:
- Probe Accuracy: 0.570
- Adversarial KL Divergence: 0.016
- Intervention Effect: 0.012
- Faithfulness: 0.000

---

**Hypothesis 4**: Copies the token following a specific pattern via induction from the previous layer

**Circuit Path**: L4.H0 → L5.H5

**Verdict**: REJECTED (Overall Score: 0.224)

**Validation Metrics**:
- Probe Accuracy: 0.600
- Adversarial KL Divergence: 0.051
- Intervention Effect: 0.005
- Faithfulness: 0.000

---


### Layer 10, Head 7

**Hypothesis 1**: Copies the token based on high induction score via self-attention

**Circuit Path**: L10.H7

**Verdict**: REJECTED (Overall Score: 0.271)

**Validation Metrics**:
- Probe Accuracy: 0.700
- Adversarial KL Divergence: 0.029
- Intervention Effect: 0.073
- Faithfulness: 0.000

---

**Hypothesis 2**: Attends to the previous token and then uses induction to predict the next token

**Circuit Path**: L9.H3 → L10.H7

**Verdict**: REJECTED (Overall Score: 0.244)

**Validation Metrics**:
- Probe Accuracy: 0.641
- Adversarial KL Divergence: 0.029
- Intervention Effect: 0.050
- Faithfulness: 0.000

---

**Hypothesis 3**: Attends to the current token based on its positional information

**Circuit Path**: L10.H7

**Verdict**: REJECTED (Overall Score: 0.184)

**Validation Metrics**:
- Probe Accuracy: 0.500
- Adversarial KL Divergence: 0.009
- Intervention Effect: 0.026
- Faithfulness: 0.000

---

**Hypothesis 4**: Copies the token following a specific pattern via induction from the previous layer

**Circuit Path**: L9.H3 → L10.H7

**Verdict**: REJECTED (Overall Score: 0.214)

**Validation Metrics**:
- Probe Accuracy: 0.567
- Adversarial KL Divergence: 0.020
- Intervention Effect: 0.043
- Faithfulness: 0.000

---


### Layer 5, Head 1

**Hypothesis 1**: Directly attends to the current token with minimal contextual influence

**Circuit Path**: L5.H1

**Verdict**: REJECTED (Overall Score: 0.294)

**Validation Metrics**:
- Probe Accuracy: 0.733
- Adversarial KL Divergence: 0.032
- Intervention Effect: 0.118
- Faithfulness: 0.000

---

**Hypothesis 2**: Captures local sequence patterns through a simple feed-forward attention flow

**Circuit Path**: L4.H0 → L5.H1

**Verdict**: REJECTED (Overall Score: 0.275)

**Validation Metrics**:
- Probe Accuracy: 0.700
- Adversarial KL Divergence: 0.026
- Intervention Effect: 0.095
- Faithfulness: 0.000

---

**Hypothesis 3**: Attends to the current token and its positional context to inform the induction process

**Circuit Path**: L5.H1

**Verdict**: WEAK (Overall Score: 0.339)

**Validation Metrics**:
- Probe Accuracy: 0.933
- Adversarial KL Divergence: 0.023
- Intervention Effect: 0.027
- Faithfulness: 0.000

---

**Hypothesis 4**: Combines information from the previous layer's representation with the current layer's induction head to capture positional relationships

**Circuit Path**: L4.H0 → L5.H1

**Verdict**: REJECTED (Overall Score: 0.283)

**Validation Metrics**:
- Probe Accuracy: 0.733
- Adversarial KL Divergence: 0.023
- Intervention Effect: 0.082
- Faithfulness: 0.000

---


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
