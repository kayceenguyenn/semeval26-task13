# 🔬 Research Direction

## Primary Research Question

**Can we reliably distinguish AI-generated code from human-written code using machine learning?**

### Sub-Questions
1. What features best discriminate between human and AI code?
2. Do different AI models have distinct coding signatures?
3. Can hybrid (human-edited AI) code be detected?
4. How do models generalize across programming languages?

---

## Research Hypotheses

### H1: Feature Engineering
- **Hypothesis:** Hand-crafted features (AST, complexity, style) can achieve 65-75% F1
- **Rationale:** AI-generated code may have distinct patterns in structure, naming, and complexity
- **Test:** Implement 100+ features, measure individual and combined impact

### H2: Transformer Fine-tuning
- **Hypothesis:** Fine-tuned CodeBERT can achieve 85-95% F1
- **Rationale:** Pre-trained code models capture semantic patterns
- **Test:** Fine-tune CodeBERT, GraphCodeBERT, StarCoder on our dataset

### H3: Hybrid Approaches
- **Hypothesis:** Combining transformers + hand-crafted features outperforms either alone
- **Rationale:** Complementary signal - transformers capture semantics, features capture style
- **Test:** Late fusion, early fusion, and attention-based fusion

### H4: Transfer Learning
- **Hypothesis:** Models trained on Task A transfer to Tasks B and C
- **Rationale:** Detection capability is foundational to classification
- **Test:** Multi-task learning, progressive fine-tuning

---

## Research Roadmap

```mermaid
gantt
    title SemEval 2026 Task 13 - Research Timeline
    dateFormat YYYY-MM-DD
    
    section Official Dates
    Evaluation Start         :milestone, 2026-01-10, 0d
    Evaluation End           :crit, milestone, 2026-01-24, 0d
    Paper Submission         :milestone, 2026-02-28, 0d
    SemEval Workshop         :milestone, 2026-07-01, 0d
    
    section Phase 1: Foundation (Nov-Dec 2025)
    Pipeline Setup           :done, 2025-11-12, 1d
    Student Onboarding      :active, 2025-11-13, 7d
    Basic Features          :2025-11-20, 14d
    Baseline Models         :2025-12-04, 14d
    
    section Phase 2: Advanced Features (Dec 2025)
    AST Features            :2025-12-18, 14d
    Complexity Metrics      :2025-12-25, 14d
    Feature Engineering     :2026-01-01, 9d
    
    section Phase 3: Competition (Jan 2026)
    Transformer Setup       :2026-01-01, 9d
    Test Data Released      :milestone, 2026-01-10, 0d
    Model Optimization      :crit, 2026-01-10, 10d
    Final Predictions       :crit, 2026-01-20, 4d
    Final Submission        :crit, milestone, 2026-01-24, 1d
    
    section Phase 4: Paper (Feb 2026)
    Results Analysis        :2026-01-25, 7d
    Paper Writing           :2026-02-01, 27d
    Paper Submission        :crit, milestone, 2026-02-28, 1d
```

---

## Performance Milestones

```mermaid
graph LR
    A[Baseline<br/>50-60% F1] -->|+10-15%| B[AST Features<br/>65-75% F1]
    B -->|+10-15%| C[Transformers<br/>85-90% F1]
    C -->|+5%| D[Optimized<br/>90-95% F1]
    
    style A fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#000
    style B fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000
    style C fill:#c5e1a5,stroke:#558b2f,stroke-width:2px,color:#000
    style D fill:#66bb6a,stroke:#2e7d32,stroke-width:3px,color:#000
```

---

## Experimental Plan

### Experiment 1: Feature Ablation Study
- **Goal:** Identify most discriminative features
- **Method:**
  1. Implement 100+ features across categories
  2. Train model with all features
  3. Remove feature groups one at a time
  4. Measure F1 drop to identify importance
- **Expected Outcome:** 5-10 critical features account for 80% of performance

### Experiment 2: Transformer Comparison
- **Goal:** Find best pre-trained model for code detection
- **Method:**
  1. Fine-tune CodeBERT, GraphCodeBERT, CodeT5, StarCoder
  2. Same hyperparameters for fair comparison
  3. Evaluate on validation set
- **Expected Outcome:** GraphCodeBERT performs best due to data flow awareness

### Experiment 3: Hybrid Architecture
- **Goal:** Combine transformers + features optimally
- **Method:**
  1. Late fusion: Average predictions
  2. Early fusion: Concatenate embeddings
  3. Attention fusion: Learn feature weights
- **Expected Outcome:** +2-5% F1 improvement from hybrid approach

### Experiment 4: Data Augmentation
- **Goal:** Increase training data diversity
- **Method:**
  1. Code transformations (variable renaming, formatting changes)
  2. Train with augmented data
  3. Measure generalization on test set
- **Expected Outcome:** +3-7% F1 improvement

---

## Expected Outcomes

| Approach | Expected F1 | Timeline | Difficulty |
|----------|-------------|----------|------------|
| Baseline (Random Forest) | 50-60% | ✅ Done | ⭐ Beginner |
| + Basic Features | 60-65% | Nov 2025 | ⭐ Beginner |
| + AST Features | 65-75% | Dec 2025 | ⭐⭐ Intermediate |
| + Complexity Metrics | 70-80% | Dec 2025 | ⭐⭐ Intermediate |
| CodeBERT Fine-tuned | 85-95% | Jan 2026 | ⭐⭐⭐ Advanced |
| Ensemble + Optimization | 90-95% | Jan 2026 | ⭐⭐⭐ Advanced |

---

## Success Criteria

**Minimum Viable:**
- ✅ 60%+ F1 on Task A (better than random)
- ✅ Working pipeline
- ✅ Reproducible results

**Target:**
- 🎯 70%+ F1 on Task A (competitive)
- 🎯 Published paper at SemEval 2026
- 🎯 Novel insights into AI code detection

**Stretch:**
- 🚀 90%+ F1 on Task A (top-tier)
- 🚀 Generalization across all 3 tasks
- 🚀 State-of-the-art performance

---

For implementation details, see [README.md](README.md)
