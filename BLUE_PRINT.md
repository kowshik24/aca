Paper Blueprint for **ACAR (Amortized Causal Abstraction with Compositional Rewrite Rules)**. It covers motivation, formal definitions, architecture, experiments, and a timeline — all aligned with the expectations of ICML/NeurIPS.

---

## Title

**Amortized Causal Abstraction: Discovering Compositional Neural Mechanisms for Multi‑Hop Reasoning in LLMs**

---

## Abstract (Draft)

We introduce **Amortized Causal Abstraction (ACA)**, a framework that automatically learns compositional, human-interpretable rules that govern how large language models (LLMs) implement multi‑step reasoning. Current mechanistic interpretability methods discover one circuit per task and cannot scale to novel task compositions. ACA addresses this by training a hypernetwork that, for any given symbolic reasoning program, predicts a set of activation interventions that causally align the LLM’s internal computation with that program. Once trained, the hypernetwork reveals a finite set of **compositional rewrite rules** — reusable neural primitives that the model combines to solve entire task families. We demonstrate ACA on multi‑hop relational inference, where a 7B‑parameter LLM trained on short chains generalises to longer chains, unseen relations, and novel query types. Our learned rules faithfully predict the model’s output under counterfactual interventions, achieving over 90% faithfulness on held‑out compositions, and we show that rule application captures the information‑theoretic sufficiency of the model’s reasoning. This work provides the first scalable, causally‑grounded framework for explaining emergent compositional behaviour in transformers.

---

## 1. Introduction

- **Problem**: Despite impressive capabilities, LLMs remain black boxes. We need explanations that are **faithful** (causally influence output), **scalable** (apply to many tasks), and **compositional** (explain how known primitives combine to produce novel behaviour).
- **Current gap**: Circuit discovery yields per‑task circuits but doesn’t tell us *how the model re‑uses those circuits across tasks*. Post‑hoc attributions lack causal guarantees. Self‑explanations are often unfaithful.
- **Our insight**: If we can learn a mapping from symbolic reasoning steps to the model’s internal interventions, we can extract **reusable rewrite rules** that explain any composition of those steps.
- **Main contributions**:
    1. Formalisation of Amortized Causal Abstraction — a novel combination of IIT and hypernetwork‑based amortization.
    2. A method to extract a finite set of compositional rewrite rules from the hypernetwork.
    3. A systematic empirical demonstration on multi‑hop relational reasoning, showing generalisation to unseen chain lengths, relations, and query types with state‑of‑the‑art faithfulness.
    4. An information‑theoretic evaluation (Effective Mutual Information) that quantifies explanation sufficiency.

---

## 2. Related Work

| Topic | Key References | How we differ |
|-------|---------------|---------------|
| **Mechanistic Interpretability** | Circuit discovery (Conmy et al., Marks et al.), SAEs, singular decomposition | We go beyond static circuits; we extract reusable compositional rules that amortise across tasks. |
| **Causal Abstraction** | Geiger et al. (IIT), Wu et al. | We extend IIT to *amortized* alignment via hypernetwork, enabling rule extraction. |
| **Feature Attribution** | SHAP, Integrated Gradients, TokenSHAP, llmSHAP | Those provide token importance, not mechanistic insight into compositional computation. |
| **Self‑Explanations** | Chain‑of‑Thought, Faithfulness benchmarks (FEEval, FaithLM) | We provide causal, neural grounding rather than post‑hoc natural language. |
| **Modularity & Reusable Components** | Probing, causal mediation analysis | We learn an explicit mapping from symbolic primitives to neural interventions, revealing modularity. |
| **Information‑theoretic XAI** | EMI (ICML 2025), Mutual Information probes | We use EMI only as an evaluation tool to measure sufficiency, not as an explanation generation method. |

---

## 3. Preliminaries and Formal Framework

### 3.1 Causal Abstraction
Given a high‑level causal model \(\mathcal{H}\) (a symbolic program) and a neural model \(\mathcal{N}\), an alignment is a set of interventions that make \(\mathcal{N}\) simulate \(\mathcal{H}\) under interchange interventions (IIT). \(\mathcal{H}\) operates on abstract variables \(V\); the alignment maps each \(V\) to a subspace of \(\mathcal{N}\)’s activations.

### 3.2 The DSL \(\mathcal{L}\) and Program Space
We define a domain‑specific language \(\mathcal{L}\) with a small set of primitive functions \(\mathcal{F} = \{f_1, \dots, f_K\}\) (e.g., `EXPAND_SET`, `FILTER`). A program \(P\) is a DAG of primitive calls. Each primitive has a well‑defined input/output type and a deterministic symbolic implementation. The set of all programs over the chosen task family is \(\mathcal{P}\).

### 3.3 Amortized Causal Abstraction
Instead of learning one alignment per program, we learn a **hypernetwork** \(\Phi_\theta\) that, given a representation of a program \(P\) (a sequence of primitive calls), outputs a set of intervention parameters. Specifically, for each step \(t\) of the program, \(\Phi_\theta\) predicts:
- A layer \(l\) and token position(s) to read the abstract variable from,
- An affine transformation (gain + bias) to inject the desired abstract value into the residual stream at a specific layer \(l'\) and position.

During IIT, we train \(\Phi_\theta\) to minimise the divergence between the model’s output under the predicted interventions and the ground‑truth symbolic output.

**Training objective** (IIT loss):
\[
\mathcal{L}_{IIT}(\theta) = \mathbb{E}_{P\sim\mathcal{P}}\left[ D_{\mathrm{KL}}\big( p_{\mathcal{N}}(\cdot | \mathrm{intervene}(x, \Phi_\theta(P))) \,\|\, p_{\mathcal{H}}(\cdot|P) \big) \right]
\]
where \(\mathrm{intervene}\) applies the predicted activation patches to the neural model.

### 3.4 Extracting Compositional Rewrite Rules
After training, we cluster the hypernetwork’s outputs for each primitive \(f_i\) across all programs. A cluster corresponds to a **rewrite rule** of the form:
> “Whenever the high‑level program requires primitive \(f_i\), the neural model computes it via a specific circuit \(C_i\) (a pattern of attention heads/MLP layers and their interactions).”

Formally, we extract:
\[
R_{f_i} = \{ (l_{\text{read}}, l_{\text{write}}, \text{head indices}, \text{transformation type}) \}
\]
This yields a **causal graph** that can be reused for any future program containing \(f_i\).

---

## 4. Task Family and DSL Design

### 4.1 Dataset: Multi‑Hop Relational Inference
We construct synthetic knowledge graphs of entities and relations. Each example = **context** (list of factual triples) + **question** (requiring compositional inference) + **gold answer**.
We generate five levels of difficulty:

| Level | Description | Compositional depth |
|-------|-------------|---------------------|
| L1 | Direct lookup | 1 hop |
| L2 | Transitive chain | 2–3 hops |
| L3 | Argmax/min | k hops (k = #entities‑1) |
| L4 | Multi‑constraint | Intersection of two chains |
| L5 | Cross‑relation + negation | Multi‑pipeline + complement |

**Train/Test Splits:**
- **IID**: same distributions as training.
- **Length generalisation**: train on L1–L3 (≤3 hops), test on L4–L5 (4–6 hops).
- **Relation generalisation**: train on `taller`, `older` relations; test on new relation `heavier` (with fresh entity names).
- **Primitive novel composition**: train on programs that use all primitives individually but not in certain combinations (e.g., `FILTER` + `ARGMAX`); test on those combinations.

### 4.2 DSL Primitives (symbolic functions)
We define 7 primitives:

1. `LOAD_CONTEXT(text) → Set[Triple]`
2. `QUERY_1HOP(entity, relation, direction) → Set[Entity]`
3. `EXPAND_SET(Set[Entity], relation, direction) → Set[Entity]`
4. `FILTER(Set[Entity], relation, direction, target) → Set[Entity]`
5. `ARGMAX/MIN(Set[Entity], relation, direction) → Entity`
6. `INTERSECT(Set1, Set2) → Set[Entity]`
7. `OUTPUT(entity) → String`

Each primitive is deterministic and fully specified (cf. pseudo‑code in appendix).

---

## 5. Architecture and Training Details

### 5.1 Base LLM
We use an open‑source, decoder‑only model: **OLMo‑7B** (or Llama‑3‑8B). We access all activation tensors. The model is frozen during hypernetwork training.

### 5.2 Hypernetwork \(\Phi_\theta\)
- Input: A sequence of primitive calls. Each call is encoded as a tuple: `(primitive_id, inputs, outputs)` where inputs/outputs are entity/relation embeddings (we use a small learnable embedding or one‑hot vectors for entities/relations; or we can leverage the LLM’s own token embeddings for entity names).
- Architecture: A shallow Transformer encoder (2 layers, 4 heads) that processes the sequence of primitive calls to produce a context‑aware representation for each step. Then an MLP decodes into intervention parameters.
- Output: For each primitive call, \(\Phi_\theta\) predicts:
  - `layer_read`: which layer’s residual stream to read the input set(s) from (usually a late layer after the context is processed).
  - `layer_write`: which layer to inject the output set into (often an intermediate layer for the next primitive).
  - `position_read/write`: a positional mask (e.g., positions corresponding to entity names, or a special `[SEP]` token).
  - `transform`: a linear projection matrix and bias that maps the abstract representation of the output set to a vector to add to the residual stream. (We can constrain this to a low‑rank adaptation or a simple vector for simplicity.)
- **Amortization**: The same hypernetwork is used for all programs; it must learn to generate correct interventions for any program.

### 5.3 Training Procedure
1. **Generate training programs**: For each example, produce the symbolic trace.
2. **Intervention execution**: For a given program \(P\), we run the LLM on the raw text input, capture all activations. Then we apply the hypernetwork’s predicted interventions: we replace (or add to) the activations at the specified layers/positions with the affine‑transformed abstract values. The abstract values are obtained by symbolically executing the primitive (i.e., we can compute the ground‑truth set of entities and encode it as a vector via a learned inverse mapping).
3. **Loss**: KL divergence between the model’s output logits after interventions and the gold answer distribution (one‑hot). Optionally, add an auxiliary loss to ensure the intervened model’s intermediate representations preserve the abstract variable values (alignment loss).
4. **Training iterations**: We train the hypernetwork until IIT loss converges. Use AdamW with a small learning rate. Batch size 128 programs.

### 5.4 Rule Extraction
After training, we feed a large set of programs covering all primitives and cluster the predicted interventions. For each primitive \(f_i\), we compute the mode of the predicted `(layer_read, layer_write, head pattern)` across examples. The mode becomes the rewrite rule.

---

## 6. Experiments and Results

### 6.1 Evaluation Metrics
- **Faithfulness (Counterfactual Success Rate)**: For a test program, we apply the extracted rule and intervene by either:
  - *Inserting* the correct abstract value (should not change the model’s correct prediction).
  - *Corrupting* the abstract value (should change the output as predicted by the symbolic program).
  We measure the percentage of interventions where the output changes exactly as expected.
- **Generalisation**: Faithfulness on unseen chain lengths, new relations, and novel primitive combinations.
- **Sufficiency (Information‑theoretic)**: Using Effective Mutual Information (EMI) between the intervened representations and the model’s output, we quantify how much of the model’s predictive uncertainty is captured by the causal graph.
- **Plausibility**: Human evaluation? (Optional; we can rely on counterfactual metrics as ICML prefers objective benchmarks.)
- **Baselines**:
  - **Static IIT** (per‑task alignment, no amortization)
  - **Circuit discovery** (ACDC) applied to the same tasks
  - **Feature attribution** (Integrated Gradients, TokenSHAP) — measure how well attributions align with the primitives.
  - **LLM self‑explanations** (CoT): measure faithfulness via stance consistency.

### 6.2 Results Table (Expected)
| Method | IID Faithfulness | Length Gen. | Relation Gen. | Primitive Novel Comb. | Sufficiency (EMI) |
|--------|------------------|-------------|---------------|----------------------|-------------------|
| ACAR (ours) | 95% | 91% | 93% | 89% | 0.82 |
| Static IIT | 93% | 62% | 75% | 45% | 0.70 |
| ACDC circuit | 80% | 30% | 50% | 20% | 0.55 |
| TokenSHAP | — | — | — | — | — (not causal) |
| CoT faithfulness (RFEval) | 70% | 65% | 68% | 60% | 0.50 |

**Key finding**: ACAR maintains high faithfulness even under compositional generalisation, while baselines collapse, demonstrating that the learned rewrite rules capture reusable neural primitives.

### 6.3 Ablation Studies
- **Without hypernetwork** (direct optimisation of per‑task alignment) — fails to generalise.
- **No amortisation** (trained on one task family only) — limited to seen compositions.
- **Size of hypernetwork** — minimal impact beyond 2‑layer transformer.
- **Number of training programs** — data efficiency analysis.
- **Impact of DSL granularity** — using a finer‑grain DSL (more primitives) improves generalisation but increases complexity.

### 6.4 Qualitative Analysis
Show examples of discovered rewrite rules:
```
EXPAND_SET(reverse) → L9H12 reads from "source entity" positions, computes relation traversal, writes output to positions of found entities at layer 12.
FILTER → L14H2 checks if target entity is in the set, writes a boolean flag at the query token position.
```
Visualise the causal graph for a novel program, confirming that the rules compose as expected.

---

## 7. Discussion

- **Broader impact**: ACAR provides a principled way to audit LLM reasoning for safety and fairness, as it extracts the actual causal mechanisms, not just plausible stories.
- **Limitations**: Currently requires a known DSL and symbolic traces for training; extending to natural language tasks without explicit symbolic programs is a future direction. We assume the model has learned the primitives; ACAR only discovers how they are implemented. Scalability to 70B+ models requires activation patching at scale (workable with activation‑addition techniques).
- **Relation to cognitive science**: The rewrite rules resemble production rules in cognitive architectures, suggesting a bridge between neural and symbolic cognition.

---

## 8. Conclusion

We presented ACAR, the first framework that automatically discovers compositional rewrite rules that causally explain how an LLM implements multi‑step reasoning. By amortising causal abstraction with a hypernetwork, we obtain a scalable and generalisable explanation that can be applied to novel task compositions. Our experiments on multi‑hop relational inference demonstrate near‑perfect faithfulness and strong generalisation, paving the way for trustworthy, mechanistic explanations of large language models.

---

## Appendix (Content to prepare)

- **Pseudo‑code for DSL primitives**.
- **Hypernetwork architecture diagram**.
- **Training algorithm pseudocode** (Alg. 1).
- **Datasheet for synthetic dataset**.
- **Details of EMI estimation** (using variational approximation).
- **Full experimental setup** (hardware, hyperparameters).

---

## Actionable Timeline for ICML 2026 Submission

- **Month 1–2**: Implement synthetic data generator and gold trace generator. Set up OLMo‑7B inference with activation patching hooks.
- **Month 3**: Design and train baseline circuits (ACDC) and static IIT. Implement hypernetwork and IIT training loop.
- **Month 4**: Run ACAR training on L1–L3 tasks. Iterate on hypernetwork architecture.
- **Month 5**: Evaluate on L4–L5 and unseen relations. Extract rules and compute faithfulness + EMI.
- **Month 6**: Run ablation studies and comparison baselines. Start paper draft.
- **Month 7**: Finalise experiments, generate plots and tables. Complete writing.
- **Month 8**: Internal review, polish, submit to NeurIPS (May deadline) or ICML (January) depending on timing.

---

This blueprint is self‑contained, concrete, and directly builds on proven techniques (IIT, mechanistic interpretability) while introducing the crucial innovation of amortised, reusable rules. It positions your work exactly at the frontier of the field. Good luck!
