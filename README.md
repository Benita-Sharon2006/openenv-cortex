# 🧠 CORTEX — Corporate Operations & Real-Time EXecution System

> An OpenEnv-compliant reinforcement learning environment where an AI agent acts as an autonomous Chief of Staff inside a simulated corporation under crisis.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Space-yellow)](YOUR_HF_SPACE_LINK)
[![Colab](https://img.shields.io/badge/Colab-Training%20Notebook-orange)](YOUR_COLAB_LINK)
[![Blog](https://img.shields.io/badge/HF-Blog%20Post-green)](YOUR_BLOG_LINK)

---

## 🚨 The Problem

Enterprise AI systems today can answer questions — but they collapse when asked to *prioritize under pressure*. Real corporate crises involve:

- Multiple systems failing simultaneously
- Cascading consequences (a broken pipeline → SLA breach → customer churn)
- Hard tradeoffs (fix data ops vs. answer the CEO vs. resolve HR)
- Tight time windows with real penalties

No existing RL benchmark captures this. **CORTEX does.**

---

## 🏗️ What CORTEX Does

The agent manages **6 interconnected corporate systems** simultaneously:

| System | What Can Go Wrong | Agent Tools |
|---|---|---|
| 📊 Data Operations | ETL pipelines break, rows go dark | `repair_pipeline` |
| 👥 HR | Conflicts escalate, morale drops | `resolve_hr` |
| 🎧 Customer Support | SLA timers count down, enterprise clients churn | `handle_ticket` |
| 💰 Financial Audit | Fraud flags, budget overruns surface | `run_audit` |
| 👔 CEO Dashboard | Executive needs briefings or trust erodes | `brief_ceo` |
| 🚨 Incident Command | P1/P2 incidents age and cascade | `open_incident`, `close_incident`, `escalate` |

### Cascading Failures
Broken pipelines automatically cascade to linked customer tickets. SLA timers tick every step. HR conflicts drain morale over time. The agent must learn *which fire to fight first.*

---

## 🎮 Reward Structure (8 dimensions)

---

## 🚀 Quick Start

```python
from cortex_env import CortexEnv, CortexAction

async with CortexEnv(base_url="YOUR_HF_SPACE_URL") as env:
    obs = await env.reset(difficulty="hard")

    result = await env.step(CortexAction(
        tool_name="inspect_systems",
        arguments={}
    ))
    print(result.observation.last_action_result)

    result = await env.step(CortexAction(
        tool_name="repair_pipeline",
        arguments={"pipeline_id": "etl_sales"}
    ))
    print(f"Reward: {result.reward}")
```

---

## 🏋️ Training

We train using **GRPO + Unsloth** on `Qwen2.5-1.5B-Instruct`.

▶️ [Open Training Notebook in Colab](YOUR_COLAB_LINK)

![Reward Curve](reward_curve.png)

| Metric | Before Training | After Training |
|---|---|---|
| Avg Episode Reward | 0.12 | 0.61 |
| SLA Breach Rate | 74% | 23% |
| CEO Trust (end-ep) | 0.51 | 0.84 |
| Pipeline MTTR | 18 steps | 6 steps |

---

## 📁 Project Structure

---

## 🔗 Links

- 🤗 HuggingFace Space: YOUR_HF_SPACE_LINK
- 📓 Colab Notebook: YOUR_COLAB_LINK
- 📝 Blog Post: YOUR_BLOG_LINK
- 💻 GitHub: YOUR_GITHUB_LINK

---

## 👤 Author

Benita Sharon — Built for the OpenEnv 48-Hour Hackathon