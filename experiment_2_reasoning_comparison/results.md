# Experiment 2: Reasoning Strategy Comparison

## Setup
- Same 4-agent misinformation scenario as Experiment 1
- All using ollama/llama3.2:3b
- 3 steps per strategy
- Strategies tested: ChainOfThought (CoT), ReAct, ReWOO

## Results

| Metric | ChainOfThought | ReAct | ReWOO |
|--------|---------------|-------|-------|
| Tool call success rate | 75.0% | 100.0% | 83.3% |
| Avg time per step | 55.2s | 83.0s | 43.7s |
| Total time | 662s | 996s | 524s |
| Most common tool | spread_rumor | spread_rumor | check_neighbors |
| Belief changes | 7 | 9 | 5 |

## Key Findings

1. ReAct achieves 100% tool call success — the most reliable strategy for tool-calling tasks
2. ReWOO is fastest (524s total vs 996s for ReAct) due to multi-step plan caching, but produces NoneType errors and has 83.3% success
3. CoT has 75% success — it sometimes produces empty plans (Plan content: None) where no tool is called
4. ReWOO replays cached tool calls in 0s on subsequent steps, but this reuse means it repeats the same (often wrong) action
5. All strategies struggle with agent ID hallucination — no strategy consistently uses real IDs from check_neighbors
6. ReAct's think-act-observe loop produces the most coherent reasoning traces, making it best suited for debugging agent behavior

## Implications for Mesa-LLM

- ReAct should be the recommended default strategy for mesa-llm, especially with local models
- The graceful degradation fallback should target ReAct first (as proposed in the GSoC proposal)
- ReWOO needs bug fixes (NoneType errors) before it can be considered production-ready
- CoT works best for simple scenarios where tool calling is not the primary concern
