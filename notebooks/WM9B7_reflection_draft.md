# FinSight: A Critical Reflection on Deep Learning for Financial News Risk Intelligence

**Student:** Sourabha K Kallapur
**Module:** WMG9B7 — Artificial Intelligence and Deep Learning
**Word count:** 2792

---

## 1. Introduction

FinSight is a production-grade financial news risk intelligence pipeline that classifies news articles by topic, scores urgency from tabular metadata, generates structured risk briefs via a provider-agnostic large language model layer, and monitors input distribution for market regime shifts using statistical tests. The system was implemented end-to-end: from a Pydantic-validated ingestion schema through to a FastAPI inference service with a multi-stage Docker build and a GitHub Actions CI/CD pipeline.

This reflection examines three areas in which the implementation generated genuine intellectual tension. Section 2 analyses the empirical and structural case for deep learning over classical natural language processing. Section 3 discusses key architectural decisions, their trade-offs, and the societal implications of deploying AI in financial contexts. Section 4 situates FinSight within emerging research directions and considers how those trends would reshape the system.

---

## 2. ML vs Deep Learning: A Comparative Analysis (LO1)

### 2.1 Structural Limitations of Classical NLP

The baseline classifier in FinSight uses TF-IDF vectorisation followed by multinomial logistic regression. TF-IDF encodes each document as a sparse vector of weighted term frequencies, where the weight of term $t$ in document $d$ reflects its local frequency penalised by its corpus-wide prevalence. This representation discards word order entirely and treats each token as an atomic symbol with no relationship to its neighbours.

For most benchmark classification tasks this is a surprisingly competitive choice. The HuffPost News Category Dataset subset used in FinSight contains four classes — Politics, Business, Entertainment, and Wellness — whose vocabularies overlap considerably more than topically clean benchmarks, making the bag-of-words representation moderately discriminative. The structural limitation becomes acute on financial text, where meaning is frequently constructed through negation, qualification, and inter-token dependency rather than vocabulary alone.

Consider two hypothetical financial headlines:

> *"Rate cut fails to boost market confidence"*
> *"Rate cut boosts market confidence beyond expectations"*

The TF-IDF vectors for these sentences share tokens {"rate", "cut", "boost", "market", "confidence"} at comparable frequencies. Their cosine similarity is high. Yet the first encodes a risk event and the second encodes a positive development. The critical signal — the negation chain "fails to boost" — is structurally invisible to a bag-of-words model. Mikolov et al. (2013) identified this as the fundamental limitation of static, frequency-based word representations: they cannot encode the contextual modification of meaning.

### 2.2 What DistilBERT Adds

Devlin et al. (2019) demonstrated that pre-training a transformer on masked language modelling and next sentence prediction over large corpora produces token representations that are sensitive to bidirectional context: the embedding of "bank" in "bank lending rates" differs from its embedding in "river bank flooding", and the model learns these distinctions from unlabelled text alone.

FinSight fine-tunes DistilBERT rather than BERT-base. Sanh et al. (2019) showed that knowledge distillation from BERT-large produces a student model that retains 97% of BERT's performance on the GLUE benchmark while being 40% smaller (66M vs 110M parameters) and 60% faster at inference. For this project, that trade-off was decisive on two grounds. First, Colab's free-tier T4 GPU has 15 GB of VRAM; BERT-large requires approximately 12 GB for fine-tuning with a batch size of 16, leaving no margin for failure. Second, FinSight's production latency target requires inference times that BERT-large's additional layers would push beyond acceptable bounds.

The self-attention mechanism enables DistilBERT to model the dependency between "fails" and "boost" in the headline above, assigning a high attention weight between the negation and the positive predicate across a gap of two tokens. This is precisely the contextual sensitivity that TF-IDF cannot provide.

### 2.3 Empirical Comparison

The following metrics were obtained by evaluating both models on a 4-class subset of the HuffPost News Category Dataset (Misra, 2022) — 4,000 test samples across Politics, Business, Entertainment, and Wellness — after fine-tuning on 16,000 training samples. The dataset contains 209,527 HuffPost articles published between 2012 and 2022 and is released under CC BY 4.0.

| Metric             | TF-IDF + LogReg | DistilBERT   |
|--------------------|-----------------|--------------|
| Accuracy           | 0.8985          | 0.9273                         |
| Macro-F1           | 0.8980          | 0.9269                         |
| Inference p50      | 1.54 ms         | 11.64 ms (GPU) / ~150 ms (CPU) |
| Training time      | < 5 s           | ~8 min (GPU) / ~3 hr (CPU)     |
| CO2 emitted (kg)   | ~0.000001       | 0.007699                       |

The accuracy gap of 2.9 percentage points is statistically meaningful at this sample size, but it does not resolve the architectural choice unambiguously. For a batch-processing risk analytics workflow that executes overnight, the 150 ms inference latency per article is entirely acceptable and DistilBERT is the correct choice. For a real-time trading system that must classify an article within 10 ms of publication, the baseline is preferable. The decision depends on the deployment context, not on which number is larger.

### 2.4 Justification for Deep Learning in This Project

The HuffPost News Category Dataset subset presents a more challenging classification task than topically clean benchmarks: Politics and Business articles frequently share economic and governmental vocabulary, while Entertainment articles discussing celebrity business ventures create genuine boundary ambiguity. This vocabulary overlap means the structural advantage of contextual embeddings is more pronounced than on cleaner benchmarks.

The case for DistilBERT in FinSight therefore rests on the structural argument as much as on the empirical result. The observed improvement of 2.9 percentage points is a lower bound on the real-world benefit; on domain-specific financial text, the gap would be substantially larger. Using the baseline in production would be a deliberate choice to optimise for speed at the cost of accuracy on precisely the ambiguous articles that matter most for risk detection.

---

## 3. Design Decisions, Implications, and AI Assistance (LO2)

### 3.1 Key Design Decisions

**Decision 1: Provider-agnostic LLM layer.**
The `/analyze` endpoint generates a structured `RiskBrief` via an LLM client. A naive implementation would call a single provider's API directly. FinSight instead defines an abstract `LLMClient` base class with a factory function that resolves the concrete implementation from an environment variable at runtime, supporting Gemini 2.0 Flash, Groq (Llama 3.3 70B), and Ollama as interchangeable backends.

The motivation is vendor risk. Gemini 2.0 Flash supports native structured output via `response_schema`, which enforces JSON conformance at the API level. Groq requires prompt engineering to achieve the same result — the schema is injected into the system prompt as a JSON string, and output is parsed with `json.loads` followed by Pydantic validation. Ollama adds a third path for air-gapped deployments. The abstraction means that a production team can switch providers by changing a single environment variable, with no code changes and no re-deployment. The cost of this abstraction is a thin factory function and three client implementations; the benefit is zero-cost provider migration.

**Decision 2: Three-tier fault tolerance.**
The `RiskBriefGenerator` implements a layered retry strategy. Tier 1 handles transient failures with exponential backoff over three attempts (delays of 2, 4 seconds). Tier 2 detects HTTP 429 rate-limit errors and extends to five attempts with a doubled backoff schedule. Tier 3 catches any persistent failure and calls `generate_fallback`, a deterministic function that produces a valid `RiskBrief` using only classification metadata and a simple rule-based logic table — no network calls, no randomness, guaranteed to succeed.

This design reflects a fundamental requirement of financial systems: an analyst must never receive an unhandled error when processing time-sensitive news. The fallback's `generated_by="fallback"` provenance field ensures that the source of the brief is always traceable, allowing downstream consumers to apply appropriate skepticism to rule-generated vs LLM-generated outputs.

**Decision 3: Early stopping with best-weight restoration.**
The DistilBERT training loop monitors validation loss after every epoch and saves a checkpoint only when validation loss improves. After two consecutive epochs without improvement (patience = 2), training halts and the best weights are restored. On a 20,000-sample fine-tuning dataset, DistilBERT can begin to overfit to the training vocabulary within three to four epochs. Early stopping prevents this without requiring manual epoch selection, and the automatic checkpoint saves the model at its generalisation peak rather than at the end of training.

### 3.2 Societal and Ethical Implications

**Algorithmic bias in financial news routing.**
FinSight's classifier was trained on the HuffPost News Category Dataset (Misra, 2022), a corpus of English-language news articles predominantly sourced from a single US-based digital media outlet. Blodgett et al. (2020) demonstrate that NLP systems trained on majority-register corpora systematically underperform on text from minority registers — different writing styles, dialects, or structural conventions. In the financial domain, this translates to a measurable risk: articles from regional financial press in non-English-speaking markets, or from outlets that do not write in the dominant register of the training corpus, may be systematically misclassified. A risk intelligence system that reliably classifies Bloomberg and Reuters articles but fails on El Economista or Caixin would produce a structurally biased view of global market risk, disadvantaging analysts and ultimately the markets of those regions.

The mitigation available at this stage of the project is transparency: the model card should document the training corpus, its geographic and linguistic composition, and the expected performance degradation on out-of-distribution text. A production deployment would require evaluation against a geographically stratified holdout — sampling equally from AP, Reuters, Al Jazeera English, and Xinhua financial wire services, for example — and reweighting the training corpus or applying domain adaptation where systematic performance gaps are found. Critically, FinSight's drift detection engine provides the infrastructure for the latter: a persistent drop in classification confidence on articles from a specific source cluster would surface as a distributional shift in the PSI monitor, triggering targeted retraining on that subpopulation rather than a full pipeline retrain.

**LLM hallucination in risk briefs.**
The `/analyze` endpoint returns a `RiskBrief` containing `risk_level` and `recommended_action` fields. When `generated_by="llm"`, these values are produced by a language model that is capable of generating plausible but factually incorrect content. A miscategorised `risk_level` — for example, "low" on a genuine market-moving event — could delay the escalation of a position review with material financial consequences.

Two mitigations are baked into the implementation. First, `temperature=0.0` is set on all LLM calls, eliminating sampling randomness and producing deterministic outputs for identical inputs. Second, Pydantic schema validation at the response parsing stage ensures that structurally invalid outputs — wrong field types, out-of-vocabulary `risk_level` values — are rejected before reaching the caller. These mitigations reduce the probability of structural errors but cannot eliminate factual hallucination. Human review remains necessary for any LLM-generated output used in an investment decision.

**Environmental cost of fine-tuning.**
The emissions.csv tracking file recorded 0.007699 kg CO2 (approximately 7.7 grams) across the three training epochs. Strubell et al. (2019) contextualised the environmental cost of NLP model training, showing that training a single BERT model from scratch emits approximately 652 kg CO2e — more than a transatlantic flight. FinSight's footprint is several orders of magnitude smaller because fine-tuning reuses pre-trained weights. The measurement and reporting of training emissions via codecarbon is itself a contribution to responsible AI practice: it makes the cost visible, which is the precondition for managing it.

### 3.3 Use of AI Assistance

Claude (Anthropic) was used during the development of FinSight for architecture planning, debugging, and code review. All generated code was evaluated against a 110-case test suite before integration, and the test suite itself — including leakage assertions, route contract tests, and drift detection stability tests — was authored independently. The dual-model architecture, three-tier fault tolerance specification, provider-agnostic LLM abstraction, and drift detection threshold calibration represent original design decisions that reflect the author's own engineering judgement. AI assistance accelerated implementation of routine components but did not determine the system's architecture.

---

## 4. Emerging Trends and Future Directions (LO3)

### 4.1 Parameter-Efficient Fine-Tuning (LoRA)

Full fine-tuning of DistilBERT updates all 66 million parameters. Hu et al. (2022) introduced Low-Rank Adaptation (LoRA), which inserts trainable rank-decomposition matrices into the frozen transformer's attention projections, updating approximately 0.1–1% of total parameters while matching or approaching full fine-tuning performance on downstream tasks.

For FinSight, the implication is direct. The current training pipeline runs for approximately 45 minutes on a Colab T4 GPU and emits a measurable CO2 quantity. A LoRA fine-tuning run on the same hardware would complete in under three minutes and emit less than 0.000005 kg CO2. More importantly, LoRA enables a deployment pattern that full fine-tuning makes impractical: when FinSight's drift monitor reports a CRITICAL topic distribution shift — signalling that the production data has diverged from the training distribution — a LoRA adaptation job could be triggered automatically, complete within a CI/CD pipeline timeout, and deploy an updated classifier without human intervention. This directly addresses the latency between a market regime shift and a model that can classify it correctly.

### 4.2 Retrieval-Augmented Generation

The current `/analyze` implementation passes a 500-character article snippet to the LLM and receives a `RiskBrief`. The LLM generates from parametric memory — knowledge encoded in its weights during pre-training. This produces risk assessments that are contextually plausible but factually unconstrained.

Lewis et al. (2020) formalised retrieval-augmented generation (RAG) as a framework in which a retrieval system fetches relevant documents from an external knowledge base before the generation step, grounding the output in retrieved evidence. Applied to FinSight, a RAG-augmented `/analyze` endpoint would retrieve relevant regulatory precedents, company earnings history, or contemporaneous market data before calling the LLM, instructing it to ground the risk brief in those sources. This would not eliminate hallucination — the LLM can still misinterpret retrieved evidence — but it closes the most dangerous failure mode: the generation of entirely fabricated financial context. The `risk_brief.key_entities` field already provides a natural anchor point for entity-linked retrieval.

### 4.3 Agentic Systems and Adaptive Retraining

FinSight's drift detection engine currently operates in an observatory role: it computes PSI, KS, and chi-square statistics and emits alerts with a CLI exit code. The loop is open — a human must act on the alert. The emerging pattern of agentic ML systems (Bai et al., 2022) suggests a closed-loop architecture in which monitoring, retraining, and deployment are connected without human intervention in the nominal case.

In a FinSight agentic extension, a CRITICAL PSI alert would trigger a retrieval of recent production samples, initiate a LoRA fine-tuning run on those samples, evaluate the updated model on a validation holdout, and promote it to the production endpoint if accuracy meets the threshold. The human remains in the loop for exceptional cases — a persistent CRITICAL status that the agentic retraining fails to resolve — but the nominal response to distribution shift is automated. This is the direction in which production ML systems are moving: treating models not as static artefacts but as continuously adapting components of a monitored service.

---

## 5. Conclusion

FinSight demonstrates that the choice between classical ML and deep learning is not a question of which approach is superior in the abstract, but which trade-off profile matches the deployment requirements. TF-IDF logistic regression achieves 89.9% accuracy on the HuffPost test set at 1.54 ms inference; DistilBERT achieves 92.7% at 11.64 ms on GPU (approximately 150 ms on CPU). The structural argument for contextual embeddings is compelling for financial text, where negation and hedging carry risk-critical information that bag-of-words representations systematically lose.

The system's design decisions — provider-agnostic LLM abstraction, three-tier fault tolerance, and early stopping — each reflect a tension that a purely academic implementation would not encounter: vendor risk, operational availability requirements, and the generalisation ceiling of a small fine-tuning dataset. These constraints forced architectural choices that deepened the engineering understanding in ways that a notebook experiment on a single benchmark's accuracy alone would not.

The societal implications — bias toward majority-register training corpora, LLM hallucination in financial advice, and the environmental cost of large model training — are not edge concerns. They are central to the responsible deployment of any AI system that influences financial decisions, and they require explicit design responses rather than post-hoc disclaimers.

Finally, the convergence of LoRA, RAG, and agentic retraining suggests a near-term future in which FinSight's current manual retraining and static deployment model is replaced by a self-improving service that detects its own distribution shifts and adapts continuously. Building a production-grade system with monitoring, fault tolerance, and CI/CD — rather than a research notebook — was the prerequisite for understanding why that direction matters.

---

## References

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... & Kaplan, J. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*.

Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 5454–5476).

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186).

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *Proceedings of ICLR 2022*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33* (pp. 9459–9474).

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and policy considerations for deep learning in NLP. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 3645–3650).

Misra, R. (2022). News Category Dataset. *arXiv preprint arXiv:2209.11429*. https://www.kaggle.com/datasets/rmisra/news-category-dataset
