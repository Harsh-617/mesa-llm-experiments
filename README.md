# Mesa-LLM Experiments

Systematic testing of mesa-llm's features across different configurations. These experiments were conducted as part of my GSoC 2026 proposal for the Mesa-LLM stabilization project.

GitHub: [Harsh-617](https://github.com/Harsh-617)

## Experiments

### Experiment 1: Model Size Impact

Compares tool-calling reliability between llama3.2:1b and llama3.2:3b.

- **1B model:** 41.7% tool call success, 23.5s/step avg
- **3B model:** 100% tool call success, 72s/step avg
- **Finding:** Minimum 3B parameters needed for reliable tool calling

### Experiment 2: Reasoning Strategy Comparison

Compares CoT, ReAct, and ReWOO strategies using llama3.2:3b.

- **ReAct:** 100% success, slowest (996s total)
- **ReWOO:** 83.3% success, fastest (524s total)
- **CoT:** 75% success, middle ground (773s total)
- **Finding:** ReAct is most reliable for tool-calling tasks

### Experiment 3: Memory Stress Test

Tests memory growth over 10 simulation steps.

- Long-term memory grows unbounded: 0 → 9,286 chars in 10 steps
- Prompt tokens grow linearly: 13 → 3,974 tokens
- Step latency increases 3.6x due to growing prompts
- **Finding:** Bounded memory with summarization is critical

## Setup

Requires: mesa, mesa-llm, matplotlib, ollama

```bash
conda activate misinfo-model
ollama pull llama3.2:1b
ollama pull llama3.2:3b
python experiment_1_model_size/run_experiment.py
python experiment_2_reasoning_comparison/run_experiment.py
python experiment_3_memory_stress/run_experiment.py
```

## Related Work

- [Mesa-LLM Misinformation Model](https://github.com/Harsh-617/mesa-llm-misinformation-model)
- Mesa-Examples PR #420: Misinformation spread example with LLM vs rule-based comparison
- Mesa-LLM contributions: PRs #89, #130, #157, #160, #194
