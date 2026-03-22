"""Experiment 1: Model Size Impact on Tool Calling

Compares how different sized LLM models (e.g. 1B vs 3B) perform
with mesa-llm's tool calling system in a misinformation simulation.
"""

import json
import time
from collections import Counter
from pathlib import Path

import mesa
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.cell_agent import BasicMovement, HasCell
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.tools.tool_decorator import tool

load_dotenv()

# ---------------------------------------------------------------------------
# Tools (same as misinformation model)
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
        """Run one reasoning step, returning metrics dict."""
        old_belief = self.belief_score
        t0 = time.time()

        tool_call_success = False
        tool_name_called = "none"

        try:
            prompt = (
                f"You are currently a {self.stance} with belief score {self.belief_score:.2f}.\n"
                f'The rumor is: "{self.model.rumor}"\n\n'
                f"You MUST follow these steps in order:\n"
                f"Step 1: Call check_neighbors to see who is nearby. Look at the agent IDs in the result.\n"
                f"Step 2: Pick ONE agent ID from the check_neighbors result. "
                f"If your belief score is above 0.5, call spread_rumor with that agent's ID. "
                f"If your belief score is 0.5 or below, call challenge_rumor with that agent's ID. "
                f"IMPORTANT: Only use agent IDs that appeared in the check_neighbors result.\n"
                f"Step 3: Call update_belief with a new score. "
                f"If you spread the rumor, increase your score slightly (add 0.05). "
                f"If you challenged it, decrease your score slightly (subtract 0.05).\n"
            )
            plan = self.reasoning.plan(prompt=prompt)
            self.apply_plan(plan)

            # Check what tools were actually called
            tool_calls = getattr(getattr(plan, "llm_plan", None), "tool_calls", None)
            if tool_calls:
                tool_call_success = True
                called_tools = {tc.function.name for tc in tool_calls}
                # Pick the most "interesting" tool called (prefer spread/challenge over check)
                for preferred in ["spread_rumor", "challenge_rumor", "update_belief", "check_neighbors"]:
                    if preferred in called_tools:
                        tool_name_called = preferred
                        break

                # Programmatic belief update fallback
                if "spread_rumor" in called_tools:
                    self.belief_score = min(1.0, self.belief_score + 0.05)
                elif "challenge_rumor" in called_tools:
                    self.belief_score = max(0.0, self.belief_score - 0.05)

                # Update stance
                if self.belief_score > 0.7:
                    self.stance = "believer"
                elif self.belief_score < 0.3:
                    self.stance = "skeptic"
                else:
                    self.stance = "neutral"

        except Exception as e:
            print(f"    ERROR: {e}")

        step_time = time.time() - t0
        belief_changed = abs(self.belief_score - old_belief) > 1e-6

        return {
            "tool_call_success": tool_call_success,
            "tool_name": tool_name_called,
            "step_time": round(step_time, 3),
            "belief_changed": belief_changed,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ExperimentModel(mesa.Model):
    def __init__(self, llm_model="ollama/llama3.2:1b"):
        super().__init__()

        self.llm_model = llm_model
        self.rumor = (
            "The town's water supply has been contaminated with dangerous "
            "chemicals from the nearby factory."
        )

        self.grid = OrthogonalMooreGrid((5, 5), capacity=1, torus=True, random=self.random)

        # 4 agents with diverse stances
        agent_configs = [
            {
                "name": "Maria",
                "persona": "A cautious schoolteacher who values evidence and critical thinking. You don't believe things easily.",
                "initial_stance": "skeptic",
                "initial_belief": 0.2,
            },
            {
                "name": "Carlos",
                "persona": "An anxious shopkeeper who worries about health risks. You tend to believe warnings about safety.",
                "initial_stance": "believer",
                "initial_belief": 0.8,
            },
            {
                "name": "Aisha",
                "persona": "A community health worker who has seen real contamination cases before. You take such claims seriously but want proof.",
                "initial_stance": "neutral",
                "initial_belief": 0.5,
            },
            {
                "name": "Tom",
                "persona": "A local journalist always looking for a story. You're curious but need verification before reporting.",
                "initial_stance": "neutral",
                "initial_belief": 0.45,
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
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(model_name: str, num_steps: int = 3) -> list[dict]:
    """Run the misinformation model with the given LLM and collect metrics."""
    short_name = model_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Running {short_name}...")
    print(f"{'='*60}")

    model = ExperimentModel(llm_model=model_name)
    metrics = []

    for step_num in range(1, num_steps + 1):
        for agent in model.agents:
            print(f"  Step {step_num}, Agent {agent.name}...", flush=True)
            result = agent.step()
            result["model"] = model_name
            result["step"] = step_num
            result["agent"] = agent.name
            metrics.append(result)
            print(
                f"    -> tool_success={result['tool_call_success']}, "
                f"tool={result['tool_name']}, "
                f"time={result['step_time']}s, "
                f"belief_changed={result['belief_changed']}"
            )

    return metrics


def print_comparison(all_metrics: list[dict]):
    """Print a comparison table of metrics across models."""
    models = sorted(set(m["model"] for m in all_metrics))

    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Metric':<30}"
    for model_name in models:
        short = model_name.split("/")[-1]
        header += f"  {short:>16}"
    print(header)
    print("-" * 70)

    for model_name in models:
        mm = [m for m in all_metrics if m["model"] == model_name]
        short = model_name.split("/")[-1]

        # Compute per-model stats
        success_rate = sum(1 for m in mm if m["tool_call_success"]) / len(mm) * 100
        avg_time = sum(m["step_time"] for m in mm) / len(mm)
        total_time = sum(m["step_time"] for m in mm)
        tool_counts = Counter(m["tool_name"] for m in mm if m["tool_name"] != "none")
        most_common = tool_counts.most_common(1)[0][0] if tool_counts else "none"
        belief_changes = sum(1 for m in mm if m["belief_changed"])

        # Store for table printing
        if model_name == models[0]:
            rows = {}
        rows_key = model_name
        if "Tool call success rate (%)" not in (rows if model_name != models[0] else {}):
            pass  # will build below

    # Build rows properly
    row_data = {}
    for model_name in models:
        mm = [m for m in all_metrics if m["model"] == model_name]
        success_rate = sum(1 for m in mm if m["tool_call_success"]) / len(mm) * 100
        avg_time = sum(m["step_time"] for m in mm) / len(mm)
        total_time = sum(m["step_time"] for m in mm)
        tool_counts = Counter(m["tool_name"] for m in mm if m["tool_name"] != "none")
        most_common = tool_counts.most_common(1)[0][0] if tool_counts else "none"
        belief_changes = sum(1 for m in mm if m["belief_changed"])
        row_data[model_name] = {
            "Tool call success rate (%)": f"{success_rate:.1f}%",
            "Avg time per step (s)": f"{avg_time:.2f}",
            "Total time (s)": f"{total_time:.2f}",
            "Most common tool": most_common,
            "Belief changes": str(belief_changes),
        }

    for metric_name in ["Tool call success rate (%)", "Avg time per step (s)",
                        "Total time (s)", "Most common tool", "Belief changes"]:
        line = f"{metric_name:<30}"
        for model_name in models:
            line += f"  {row_data[model_name][metric_name]:>16}"
        print(line)

    print(f"{'='*70}")


def save_chart(all_metrics: list[dict], output_path: str):
    """Save a bar chart comparing tool call success rates."""
    models = sorted(set(m["model"] for m in all_metrics))
    short_names = [m.split("/")[-1] for m in models]
    success_rates = []

    for model_name in models:
        mm = [m for m in all_metrics if m["model"] == model_name]
        rate = sum(1 for m in mm if m["tool_call_success"]) / len(mm) * 100
        success_rates.append(rate)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(short_names, success_rates, color=["#4A90D9", "#D94A4A"])
    ax.set_ylabel("Tool Call Success Rate (%)")
    ax.set_xlabel("Model")
    ax.set_title("Experiment 1: Model Size Impact on Tool Calling")
    ax.set_ylim(0, 105)

    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nChart saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    # Run both models
    for model_name in ["ollama/llama3.2:1b", "ollama/llama3.2:3b"]:
        metrics = run_experiment(model_name, num_steps=3)
        all_metrics.extend(metrics)

    # Print comparison table
    print_comparison(all_metrics)

    # Save chart
    save_chart(all_metrics, str(results_dir / "model_comparison.png"))

    # Save raw metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
