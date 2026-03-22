# Experiment 3: Memory System Stress Test

## Setup
- 2 agents (Carlos/believer, Maria/skeptic) on 3x3 grid
- ollama/llama3.2:3b with ReAct reasoning
- 10 steps — testing memory growth over extended simulation

## Results

| Metric | Step 1 | Step 5 | Step 10 |
|--------|--------|--------|---------|
| Carlos STM count | 1 | 5 | 8 |
| Carlos LTM chars | 0 | 0 | 4,920 |
| Carlos prompt tokens | 13 | 988 | 2,868 |
| Carlos step time | 92s | 176s | 273s |
| Maria STM count | 1 | 5 | 8 |
| Maria LTM chars | 0 | 0 | 9,286 |
| Maria prompt tokens | 13 | 765 | 3,974 |
| Maria step time | 98s | 149s | 340s |

## Key Findings

1. Long-term memory grows without bound — Maria's LTM reached 9,286 characters in just 10 steps with no truncation or summarization
2. Prompt token count grows linearly with steps: 13 tokens at step 1 to ~4,000 at step 10. At this rate, a 30-step simulation would exceed most small model context windows (4K-8K tokens)
3. Step latency increases proportionally with memory size: step 1 took ~95s, step 10 took ~340s — a 3.6x slowdown due to larger prompts
4. Short-term memory appears capped at 8 entries (deque eviction working), but long-term memory has no such protection
5. Communication history remained at 0 throughout — agents called spread_rumor/challenge_rumor with hallucinated IDs so no actual messages were stored, meaning the memory growth would be even worse in a working simulation
6. 3 threshold warnings triggered during the run, confirming the need for bounded memory

## Implications for Mesa-LLM

- Bounded long-term memory with configurable max_tokens and automatic summarization (proposed in GSoC Section 2.3) is critical for any simulation beyond ~10 steps
- Without memory bounds, mesa-llm simulations will slow down progressively and eventually crash when prompt exceeds context window
- Memory consolidation currently uses a hard-coded openai/gpt-4o-mini model — users on Ollama will hit errors when consolidation triggers
- The linear prompt growth pattern suggests a summarize-then-store approach could reduce prompt size by 60-80% while preserving key information
