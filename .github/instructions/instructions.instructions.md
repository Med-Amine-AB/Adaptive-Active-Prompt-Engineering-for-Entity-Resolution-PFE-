# Adaptive Active Prompt Engineering for Entity Resolution (PFE)

## Project Context
This repository contains the implementation of a research-oriented but industry-focused project on **Entity Resolution (Entity Matching)** using **Large Language Models (LLMs)**.

The goal is to determine whether two textual records refer to the same real-world entity in **noisy and heterogeneous tabular datasets**, using LLMs as decision engines instead of traditional fine-tuned classifiers.

This is a **Master (Research) final project**, but the implementation should remain **practical, applied, and reproducible**, suitable for real-world data science workflows.

---

## Core Problem
Entity Resolution (ER) is the task of deciding whether two entity descriptions (records) refer to the same real-world entity.

Example:
- Record A: "Apple Inc., Cupertino, USA"
- Record B: "Apple Incorporated, CA"

Output: SAME / DIFFERENT

---

## Project Objectives
1. Use **LLMs** to perform entity matching via **prompt-based reasoning**
2. Avoid heavy fine-tuning when possible (zero-shot / few-shot focus)
3. Improve robustness using a **lightweight active learning strategy**
4. Adapt prompts iteratively based on **low-confidence / ambiguous cases**
5. Provide **explainability** by extracting short textual rationales from the LLM
6. Compare performance against **classical ER baselines**

---

## Scope Constraints (IMPORTANT)
- **Text-only entity resolution** (no images, no multimodal models)
- No theoretical proofs or heavy math
- No training large models from scratch
- Focus on **empirical evaluation and system behavior**
- Explainability should be **natural language**, not SHAP/LIME

---

## Expected System Components
1. **Data Loader**
   - Load ER datasets (pairs of records + labels)

2. **Baseline ER Methods**
   - TF-IDF + cosine similarity
   - Sentence-BERT embeddings + similarity threshold
   - Simple ML classifier (optional)

3. **LLM-based Entity Matcher**
   - Prompt-based SAME / DIFFERENT prediction
   - Zero-shot and few-shot settings
   - Structured output (label + explanation)

4. **Confidence Estimation**
   - Based on probabilities, logprobs, or consistency checks
   - Used to detect ambiguous cases

5. **Active Learning Loop**
   - Select low-confidence examples
   - Use them to refine prompts or in-context examples
   - No complex AL theory required

6. **Evaluation**
   - Precision / Recall / F1
