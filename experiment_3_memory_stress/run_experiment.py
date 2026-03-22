"""Experiment 3: Memory Stress Test

Runs a 2-agent misinformation scenario for 10 steps using ollama/llama3.2:3b
and tracks how mesa-llm's STLTMemory system grows over time — short-term count,
long-term length, communication history size, estimated prompt tokens, and
per-step wall-clock time.
"""

import json
import time
from pathlib import Path

import mesa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.cell_agent import BasicMovement, HasCell
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.tools.tool_decorator import tool

load_dotenv()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def check_neighbors(agent):
    """Check who is nearby and what they believe about the rumor.

    Returns a formatted list of neighboring agents with their current stance
    on the rumor (believer, skeptic, or neutral).
    """
    neighbors = []
    for cell in agent.cell.neighborhood:
        for neighbor in cell.agents:
            if neighbor is not agent:
                neighbors.append(neighbor)

    if not neighbors:
        return "No neighbors nearby."

    lines = []
    for n in neighbors:
        lines.append(f"- Agent {n.unique_id} ({n.name}): {n.stance}")
    return "Nearby agents:\n" + "\n".join(lines)


@tool
def spread_rumor(agent, target_id: int):
    """Spread the rumor to a specific agent to try to convince them.

    Args:
        target_id: The unique_id of the agent to spread the rumor to.
    """
    target = None
    for a in agent.model.agents:
        if a.unique_id == target_id:
            target = a
            break

    if target is None:
        return f"Could not find agent with id {target_id}."

    message = f"Have you heard? {agent.model.rumor}"
    agent.send_message(message, [target])
    return f"Spread the rumor to Agent {target_id} ({target.name})."


@tool
def challenge_rumor(agent, target_id: int):
    """Challenge the rumor by sending a counter-argument to a specific agent.

    Args:
        target_id: The unique_id of the agent to send the counter-argument to.
    """
    target = None
    for a in agent.model.agents:
        if a.unique_id == target_id:
            target = a
            break

    if target is None:
        return f"Could not find agent with id {target_id}."

    message = f"I don't believe the rumor: {agent.model.rumor}. I think it's false."
    agent.send_message(message, [target])
    return f"Challenged the rumor with Agent {target_id} ({target.name})."


@tool
def update_belief(agent, new_score: float):
    """Update personal belief score and stance based on new information.

    Args:
        new_score: The new belief score between 0.0 (disbelief) and 1.0 (full belief).
    """
    try:
        new_score = float(new_score)
    except (ValueError, TypeError):
        return f"Error: new_score must be a number, got {new_score}"

    clamped = max(0.0, min(1.0, new_score))
    agent.belief_score = clamped

    if clamped > 0.7:
        agent.stance = "believer"
    elif clamped < 0.3:
        agent.stance = "skeptic"
    else:
        agent.stance = "neutral"

    return f"Updated belief score to {clamped:.2f}. Stance is now: {agent.stance}."


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CitizenAgent(LLMAgent, HasCell, BasicMovement):
    def __init__(self, model, name, persona, initial_stance, initial_belief):
        super().__init__(
            model=model,
            reasoning=ReActReasoning,
            llm_model=model.llm_model,
            system_prompt=f"You are {name}, a citizen in a small community. {persona}",
        )
        for tool_name in ["move_one_step", "teleport_to_location", "speak_to"]:
            self.tool_manager.tools.pop(tool_name, None)

        self.name = name
        self.stance = initial_stance
        self.belief_score = initial_belief

    def step(self):
        """Run one reasoning step, returning metrics dict with memory stats."""
        t0 = time.time()

        try:
            prompt = (
                f"You are currently a {self.stance} with belief score {self.belief_score:.2f}.\n"
                f'The rumor is: "{self.model.rumor}"\n\n'
                f"You MUST follow these steps in order:\n"
                f"Step 1: Call check_neighbors to see who is nearby.\n"
                f"Step 2: Pick ONE agent from the result. "
                f"If your belief > 0.5, call spread_rumor with their ID. "
                f"Otherwise call challenge_rumor with their ID. "
                f"You MUST communicate every turn — spreading or challenging is required.\n"
                f"Step 3: Call update_belief with a slightly adjusted score "
                f"(+0.05 if you spread, -0.05 if you challenged).\n"
            )
            plan = self.reasoning.plan(prompt=prompt)
            self.apply_plan(plan)

            # Programmatic belief update fallback
            tool_calls = getattr(getattr(plan, "llm_plan", None), "tool_calls", None)
            if tool_calls:
                called_tools = {tc.function.name for tc in tool_calls}
                if "spread_rumor" in called_tools:
                    self.belief_score = min(1.0, self.belief_score + 0.05)
                elif "challenge_rumor" in called_tools:
                    self.belief_score = max(0.0, self.belief_score - 0.05)
                if self.belief_score > 0.7:
                    self.stance = "believer"
                elif self.belief_score < 0.3:
                    self.stance = "skeptic"
                else:
                    self.stance = "neutral"

        except Exception as e:
            print(f"    ERROR: {e}")

        step_time = time.time() - t0

        # --- Collect memory stats ---
        mem = self.memory

        # Short-term memory count (it's a deque)
        stm_count = len(mem.short_term_memory)

        # Long-term memory length
        ltm = mem.long_term_memory
        if isinstance(ltm, str):
            ltm_length = len(ltm)
        elif isinstance(ltm, (list, dict)):
            ltm_length = len(ltm)
        else:
            ltm_length = 0

        # Communication history
        try:
            comm_history = mem.get_communication_history()
            comm_length = len(str(comm_history)) if comm_history else 0
        except Exception:
            comm_length = 0

        # Estimate prompt length in tokens (chars / 4)
        try:
            prompt_text = mem.get_prompt_ready()
            prompt_tokens = len(str(prompt_text)) // 4
        except Exception:
            prompt_tokens = 0

        return {
            "step_time": round(step_time, 3),
            "short_term_memory_count": stm_count,
            "long_term_memory_length": ltm_length,
            "communication_history_chars": comm_length,
            "prompt_tokens_approx": prompt_tokens,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ExperimentModel(mesa.Model):
    def __init__(self, llm_model="ollama/llama3.2:3b"):
        super().__init__()
        self.llm_model = llm_model
        self.rumor = (
            "The town's water supply has been contaminated with dangerous "
            "chemicals from the nearby factory."
        )

        self.grid = OrthogonalMooreGrid((3, 3), capacity=1, torus=True, random=self.random)

        agent_configs = [
            {
                "name": "Carlos",
                "persona": (
                    "An anxious shopkeeper who worries about health risks. "
                    "You tend to believe warnings and actively spread them. "
                    "Every turn, you MUST use spread_rumor or challenge_rumor to talk to others."
                ),
                "initial_stance": "believer",
                "initial_belief": 0.8,
            },
            {
                "name": "Maria",
                "persona": (
                    "A cautious schoolteacher who values evidence. "
                    "You are skeptical of unverified claims and push back on rumors. "
                    "Every turn, you MUST use spread_rumor or challenge_rumor to talk to others."
                ),
                "initial_stance": "skeptic",
                "initial_belief": 0.2,
            },
        ]

        for config in agent_configs:
            agent = CitizenAgent(
                model=self,
                name=config["name"],
                persona=config["persona"],
                initial_stance=config["initial_stance"],
                initial_belief=config["initial_belief"],
            )
            agent.move_to(self.grid.select_random_empty_cell())


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def save_chart(all_metrics, output_path):
    """Save a line chart showing memory growth over steps."""
    agents = sorted(set(m["agent"] for m in all_metrics))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Experiment 3: Memory Growth Over 10 Steps (ollama/llama3.2:3b)", fontsize=13)

    metric_keys = [
        ("short_term_memory_count", "Short-Term Memory Count"),
        ("long_term_memory_length", "Long-Term Memory Length (chars)"),
        ("communication_history_chars", "Communication History (chars)"),
        ("prompt_tokens_approx", "Approx Prompt Tokens"),
    ]

    for ax, (key, title) in zip(axes.flat, metric_keys):
        for agent_name in agents:
            agent_data = [m for m in all_metrics if m["agent"] == agent_name]
            steps = [m["step"] for m in agent_data]
            values = [m[key] for m in agent_data]
            ax.plot(steps, values, marker="o", label=agent_name, linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nChart saved to {output_path}")


# ---------------------------------------------------------------------------
# Threshold warnings
# ---------------------------------------------------------------------------

WARN_LTM_CHARS = 2000
WARN_PROMPT_TOKENS = 4000


def check_thresholds(metrics_entry):
    """Print warnings if memory exceeds thresholds."""
    warnings = []
    agent = metrics_entry["agent"]
    step = metrics_entry["step"]

    if metrics_entry["long_term_memory_length"] > WARN_LTM_CHARS:
        warnings.append(
            f"  WARNING [Step {step}, {agent}]: long_term_memory = "
            f"{metrics_entry['long_term_memory_length']} chars (threshold: {WARN_LTM_CHARS})"
        )

    if metrics_entry["prompt_tokens_approx"] > WARN_PROMPT_TOKENS:
        warnings.append(
            f"  WARNING [Step {step}, {agent}]: prompt_tokens ~ "
            f"{metrics_entry['prompt_tokens_approx']} (threshold: {WARN_PROMPT_TOKENS})"
        )

    for w in warnings:
        print(w)

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_STEPS = 10
    LLM_MODEL = "ollama/llama3.2:3b"

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Experiment 3: Memory Stress Test")
    print(f"Model: {LLM_MODEL}  |  Agents: 2  |  Steps: {NUM_STEPS}")
    print(f"{'='*60}")

    model = ExperimentModel(llm_model=LLM_MODEL)
    all_metrics = []
    all_warnings = []

    for step_num in range(1, NUM_STEPS + 1):
        print(f"\n--- Step {step_num} ---")
        for agent in model.agents:
            print(f"  Agent {agent.name}...", flush=True)
            result = agent.step()
            result["step"] = step_num
            result["agent"] = agent.name
            result["model"] = LLM_MODEL
            all_metrics.append(result)

            print(
                f"    STM count={result['short_term_memory_count']}, "
                f"LTM length={result['long_term_memory_length']}, "
                f"comm_chars={result['communication_history_chars']}, "
                f"prompt_tok~{result['prompt_tokens_approx']}, "
                f"time={result['step_time']}s"
            )

            step_warnings = check_thresholds(result)
            all_warnings.extend(step_warnings)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("MEMORY GROWTH SUMMARY")
    print(f"{'='*60}")

    agents = sorted(set(m["agent"] for m in all_metrics))
    for agent_name in agents:
        agent_data = [m for m in all_metrics if m["agent"] == agent_name]
        first = agent_data[0]
        last = agent_data[-1]
        print(f"\n  {agent_name}:")
        print(f"    STM count:    {first['short_term_memory_count']} -> {last['short_term_memory_count']}")
        print(f"    LTM length:   {first['long_term_memory_length']} -> {last['long_term_memory_length']}")
        print(f"    Comm chars:   {first['communication_history_chars']} -> {last['communication_history_chars']}")
        print(f"    Prompt tokens: {first['prompt_tokens_approx']} -> {last['prompt_tokens_approx']}")
        avg_time = sum(m["step_time"] for m in agent_data) / len(agent_data)
        print(f"    Avg step time: {avg_time:.2f}s")

    if all_warnings:
        print(f"\n  Total threshold warnings: {len(all_warnings)}")
    else:
        print(f"\n  No threshold warnings triggered.")

    # --- Save outputs ---
    save_chart(all_metrics, str(results_dir / "memory_growth.png"))

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
