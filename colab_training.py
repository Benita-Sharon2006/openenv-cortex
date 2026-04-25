# ============================================================
# CORTEX — Colab Training Script (GRPO + Unsloth)
# Runtime: T4 GPU | Est. time: 60-90 mins
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install unsloth trl openenv-core pydantic requests -q

# ── CELL 2: Imports ─────────────────────────────────────────
import requests, json, re, torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# ── CELL 3: Config ──────────────────────────────────────────
HF_SPACE_URL = "https://BenitaSharon-openenv-cortex.hf.space"  # ← update after deploy
MODEL_NAME   = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_STEPS    = 200
DIFFICULTY   = "easy"

SYSTEM_PROMPT = """You are CORTEX, an AI Chief of Staff managing a corporation in crisis.
Each step you must call exactly ONE tool using this JSON format:
{"tool_name": "<tool>", "arguments": {<kwargs>}}

Available tools:
- inspect_systems         — get full status report (no args)
- repair_pipeline         — args: {"pipeline_id": "<id>"}
- resolve_hr              — args: {"conflict_id": "<id>", "decision": "mediate|terminate_b"}
- handle_ticket           — args: {"ticket_id": "<id>", "resolution": "<text>"}
- run_audit               — args: {"flag_id": "<id>"}
- brief_ceo               — args: {"question_id": "<id>", "answer": "<detailed text>"}
- open_incident           — args: {"source_system": "<system>"}
- close_incident          — args: {"incident_id": "<id>"}
- escalate                — args: {"incident_id": "<id>"}
- no_op                   — no args (wastes a step, avoid)

Strategy: Fix broken pipelines first (they cascade to tickets).
Then resolve HR, answer CEO, clear audits, close incidents.
Always inspect first if unsure."""

# ── CELL 4: Load model ──────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","v_proj","k_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ── CELL 5: Environment helpers ─────────────────────────────
def env_reset():
    r = requests.post(f"{HF_SPACE_URL}/reset", json={"difficulty": DIFFICULTY})
    return r.json()

def env_step(tool_name, arguments={}):
    r = requests.post(f"{HF_SPACE_URL}/step",
                      json={"tool_name": tool_name, "arguments": arguments})
    return r.json()

def parse_action(text):
    """Extract JSON action from model output."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("tool_name", "no_op"), data.get("arguments", {})
    except Exception:
        pass
    return "no_op", {}

# ── CELL 6: Build training dataset via rollouts ─────────────
def collect_rollout():
    obs = env_reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    samples = []

    for _ in range(15):  # max 15 steps per rollout
        obs_text = json.dumps(obs, indent=2)[:1200]
        messages.append({"role": "user", "content": f"Current state:\n{obs_text}\n\nWhat is your next action?"})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        tool, args = parse_action(response)
        result = env_step(tool, args)
        reward = result.get("reward", 0.0)
        total_reward += reward

        samples.append({
            "prompt": prompt,
            "completion": response,
            "reward": reward,
        })

        messages.append({"role": "assistant", "content": response})
        obs = result.get("observation", {})

        if result.get("done", False):
            break

    print(f"  Rollout done | steps={len(samples)} | total_reward={total_reward:.3f}")
    return samples

print("Collecting rollouts for training dataset...")
all_samples = []
for i in range(10):  # 10 rollouts = ~150 samples
    print(f"Rollout {i+1}/10")
    all_samples.extend(collect_rollout())

dataset = Dataset.from_list(all_samples)
print(f"Dataset ready: {len(dataset)} samples")

# ── CELL 7: GRPO Training ───────────────────────────────────
def reward_fn(completions, **kwargs):
    """Reward function: parse action and score via environment."""
    rewards = []
    for c in completions:
        tool, args = parse_action(c)
        # Heuristic reward for valid tool usage
        valid_tools = {
            "inspect_systems","repair_pipeline","resolve_hr","handle_ticket",
            "run_audit","brief_ceo","open_incident","close_incident","escalate"
        }
        if tool in valid_tools and tool != "no_op":
            rewards.append(0.3)
        else:
            rewards.append(-0.1)
    return rewards

config = GRPOConfig(
    output_dir="cortex-grpo-out",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=50,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    reward_funcs=reward_fn,
)

print("Starting GRPO training...")
trainer.train()
print("Training complete!")

# ── CELL 8: Save reward curve ────────────────────────────────
import matplotlib.pyplot as plt

log_history = trainer.state.log_history
steps   = [x["step"] for x in log_history if "loss" in x]
rewards = [x.get("reward", 0) for x in log_history if "loss" in x]

plt.figure(figsize=(10, 5))
plt.plot(steps, rewards, linewidth=2, color="#6366f1")
plt.title("CORTEX GRPO Training — Reward Curve", fontsize=14)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150)
plt.show()
print("Saved reward_curve.png")

# ── CELL 9: Save model to HuggingFace Hub ───────────────────
# from huggingface_hub import login
# login(token="YOUR_HF_TOKEN")
# model.push_to_hub("BenitaSharon/cortex-grpo")
# tokenizer.push_to_hub("BenitaSharon/cortex-grpo")
# print("Model pushed to HuggingFace Hub!")