# Experiment 1: Model Size Impact on Mesa-LLM Tool Calling

## Setup
- 4 agents (Maria/skeptic, Carlos/believer, Aisha/neutral, Tom/neutral) on 5x5 grid
- Misinformation spread scenario with 4 custom tools
- 3 steps per run
- ReAct reasoning, built-in tools removed

## Models Tested
- ollama/llama3.2:1b (2GB, 1 billion parameters)
- ollama/llama3.2:3b (2GB, 3 billion parameters)

## Results

| Metric | llama3.2:1b | llama3.2:3b |
|--------|------------|------------|
| Tool call success rate | 41.7% | 100.0% |
| Avg time per step | 23.5s | 72.0s |
| Total time | 282s | 864s |
| Most common tool | check_neighbors | spread_rumor |
| Belief changes | 1 | 9 |

## Key Findings

1. The 3B model achieves 100% tool call success vs 41.7% for the 1B model — a 2.4x improvement in reliability
2. This comes at a 3x latency cost (72s vs 23.5s per step)
3. The 1B model can identify tools but frequently fails to format the call correctly — it outputs text like "call spread_rumor(target_id=2)" instead of structured JSON
4. The 3B model produces more belief changes (9 vs 1), indicating it actually drives simulation dynamics while the 1B model mostly observes without acting
5. Both models struggle with agent ID hallucination — calling spread_rumor with made-up IDs like 123 or 1234

## Implications for Mesa-LLM

- A minimum of 3B parameters appears necessary for reliable tool calling with mesa-llm
- The graceful degradation fallback proposed in the GSoC proposal would primarily benefit 1B-class models
- Token-per-second performance matters more than raw parameter count for simulation throughput
- Mesa-llm should document minimum model requirements for different reasoning strategies
