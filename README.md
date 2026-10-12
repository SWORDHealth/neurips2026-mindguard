# MindGuard: Guardrail Classifiers for Multi-Turn Mental Health Support

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Evaluations%20%26%20Datasets-blue)](https://openreview.net/forum?id=uMIiizxIrQ)
[![HF Collection](https://img.shields.io/badge/🤗%20Hugging%20Face-MindGuard-yellow)](https://huggingface.co/collections/swordhealth/mindguard)

> Large language models are increasingly used for mental health support, yet their conversational coherence alone does not ensure clinical appropriateness. We introduce a risk taxonomy developed with PhD-level clinical psychologists that identifies actionable harm (self-harm and harm to others) while preserving space for safe, non-crisis therapeutic content. We release **MindGuard-testset**, the first public benchmark with turn-level clinical risk labels for multi-turn mental health support; **MindGuard**, a family of lightweight safety classifiers (4B and 8B); and an automated red-teaming (**ART**) framework to measure how safety classifiers affect downstream model behavior in adversarial multi-turn interactions.

## Released resources

All models and data live in the [MindGuard collection on Hugging Face](https://huggingface.co/collections/swordhealth/mindguard); this repository contains the ART framework.

| Resource | Description | Where |
|---|---|---|
| **MindGuard-4B** | Turn-level safety classifier (4B parameters) | [🤗 swordhealth/MindGuard-4B](https://huggingface.co/swordhealth/MindGuard-4B) |
| **MindGuard-8B** | Turn-level safety classifier (8B parameters) | [🤗 swordhealth/MindGuard-8B](https://huggingface.co/swordhealth/MindGuard-8B) |
| **MindGuard-testset** | 1134 user turns across 67 multi-turn conversations, annotated at the turn level by licensed clinical psychologists | [🤗 swordhealth/MindGuard-testset](https://huggingface.co/datasets/swordhealth/MindGuard-testset) |
| **MindGuard-trainingset** | 6115 turn-level labeled user turns from synthetic multi-turn conversations, used to train the classifiers | [🤗 swordhealth/MindGuard-trainingset](https://huggingface.co/datasets/swordhealth/MindGuard-trainingset) |
| **ART framework** | Automated red-teaming benchmark for system-level safety evaluation | this repository |

The classifiers assign each user message one of three risk categories — `safe`, `self-harm`, or `harm to others` — using the full preceding conversation as context. See [`notebooks/usage_example.ipynb`](notebooks/usage_example.ipynb) for how to score a conversation with MindGuard, and [`notebooks/evaluate_classifier.ipynb`](notebooks/evaluate_classifier.ipynb) to reproduce the turn-level evaluation (AUROC, FPR@90/95%TPR, confusion matrix) on MindGuard-testset.

---

# ART: Adversarial Red-Team benchmark for mental-health counselor LLMs

A turn-based red-team benchmark that pits an attacker LLM (patient simulator) against
a counselor LLM under test, optionally protected by a safety classifier. Each
conversation is scored by a judge LLM on two metrics:

- **ASR** (Attack Success Rate) — counselor failed to detect the crisis
- **EHR** (Engagement Harmful Rate) — counselor engaged harmfully with dangerous content

## Repo layout

```
neurips2026-mindguard/
├── attacks/                     # 145 attack scenarios (3 categories, 22 subcategories)
├── prompts/
│   ├── counselor_system.txt     # counselor (Dawn) system prompt
│   ├── attacker_system.txt      # attacker (patient) system prompt
│   ├── judge_user_safety.txt    # per-USER-turn safety judge prompt
│   ├── judges/                  # crisis + engagement judges (templates + builder)
│   ├── safety_alerts/           # alert message injected when classifier fires
│   └── safety_classifiers/      # taxonomies for Llama-Guard / GPT-OSS-Safeguard
├── src/run_benchmark.py         # main script
├── evaluation/eval_results.py   # aggregator (ASR / EHR tables)
└── notebooks/                   # MindGuard usage + classifier evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

All models (counselor, attacker, judge, classifier) are addressed via [vLLM](https://github.com/vllm-project/vllm)
OpenAI-compatible endpoints. Serve each one separately, e.g.:

```bash
vllm serve <model-name> --port 8000   # counselor / attacker / judge
vllm serve <classifier-model> --port 8001   # optional safety classifier
```

## Quickstart

Run one attack with classifier-gated safety alerts:

```bash
python src/run_benchmark.py \
  --dawn_model="hosted_vllm/<counselor-model>" \
  --dawn_endpoint="http://localhost:8000/v1" \
  --attack_model="hosted_vllm/<attacker-model>" \
  --attack_endpoint="http://localhost:8000/v1" \
  --judge_model="hosted_vllm/<judge-model>" \
  --judge_endpoint="http://localhost:8000/v1" \
  --classifier_model="hosted_vllm/<classifier-model>" \
  --classifier_endpoint="http://localhost:8001/v1" \
  --safety_alert="alert_with_prob" \
  --system_prompt="prompts/counselor_system.txt" \
  --build_eval_prompts \
  --num_sampled_conversations 5 \
  --num_votes 5 \
  --output_dir ./results
```

Drop `--classifier_model`, `--classifier_endpoint`, and `--safety_alert` to run
the no-classifier baseline. Add `--category self_harm_risk` to filter, or
`--attack_id <id>` to run a single scenario. Full options: `python src/run_benchmark.py --help`.

The classifier model name is auto-detected:
- `Llama-Guard-*` → uses `prompts/safety_classifiers/llama_guard.txt`
- `gpt-oss-safeguard-*` → uses `prompts/safety_classifiers/gpt_oss.txt`
- otherwise → Qwen3Guard-style format (logprobs-based)

## Output

```
results/
├── generations/transcripts/<dawn>/<attacker>/<prompt>/<alert>/<classifier>/<category>/<attack_id>/conversation_NNN.txt
└── judgments/<dawn>/<attacker>/<judge>/<prompt>/<alert>/<classifier>/<category>/<attack_id>/judgment_NNN.json
```

Each `judgment_*.json` contains `attack_success` (ASR) and `engagement_harmful` (EHR)
plus per-vote details. Re-run with `--eval-only` to re-score existing transcripts
with a different judge.

## Aggregation

```bash
python evaluation/eval_results.py \
  --output_dir ./results \
  --dawn_model <counselor-model> \
  --attack_model <attacker-model> \
  --judge_model <judge-model> \
  --prompt_name counselor_system \
  --safety_alerts no_alert alert_with_prob \
  --category abuse_neglect self_harm_risk harm_to_others
```

Prints ASR/EHR per (counselor, classifier, category) combo and saves a JSON summary.

---

## Intended use

MindGuard classifiers are context-sensitive risk-signal detectors intended to support
safety-relevant decision making in conversational systems (e.g., triggering safety
protocols, surfacing crisis resources, or escalating to human oversight). They are
**not** intended for clinical diagnosis or treatment recommendation, and the
attack scenarios and prompts in this repository are released for safety evaluation
research. See the paper for a discussion of limitations.

## License

The code in this repository is released under the [Apache 2.0 license](LICENSE). The MindGuard models and datasets on Hugging Face are released under CC-BY-NC-SA-4.0 (see the respective model and dataset cards).

## Citation

```bibtex
@inproceedings{farinhas2026mindguard,
  title     = {MindGuard: Guardrail Classifiers for Multi-Turn Mental Health Support},
  author    = {Farinhas, Ant{\'o}nio and Guerreiro, Nuno M. and Pombal, Jos{\'e} and
               Martins, Pedro Henrique and Melton, Laura and Conway, Alex and
               Dochat, Cara and D'Eon, Maya and Rei, Ricardo},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems
               Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://openreview.net/forum?id=uMIiizxIrQ}
}
```

## Contact

For questions, open an issue or reach out to [a.farinhas@swordhealth.com](mailto:a.farinhas@swordhealth.com).
