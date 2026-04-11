# Lookahead Algorithm — Agent Context

You are working on a reinforcement learning project where a DQN agent plays a
custom Snake game (Snake+).  This document gives you the full context needed to
reason about, improve, or replace the lookahead path-selection algorithm.

---

## What the lookahead does

At inference time (evaluation / dashboard), instead of executing the DQN's
one-step greedy action directly, the lookahead:

1. Tries each of the 3 candidate actions (FORWARD, TURN_LEFT, TURN_RIGHT)
2. For each action, simulates `depth = min(snake_length, max_depth)` steps
   forward on a deepcopy of the environment using the greedy policy
3. Collects the discounted return of that simulated trajectory
4. Repeats this for `n_samples` independent samples per action (stochastic
   outcomes differ because Python's global `random` state advances between
   deepcopies — object spawn positions and timing vary)
5. Aggregates the samples into a single score per action using a
   **risk-adjusted mean** (see below)
6. Picks the action with the highest score — but only overrides the agent's
   raw prediction if the lookahead found a strictly better score

All `n_samples × n_actions` rollouts run in parallel via `ThreadPoolExecutor`.
PyTorch GPU ops (CUDA / MPS) release the GIL so threads run concurrently on
the device.

---

## Current scoring formula

```
score(action) = mean(samples) - RISK_LAMBDA * death_rate * |DEATH_SCORE_THRESHOLD|
```

- `DEATH_SCORE_THRESHOLD = -50.0` — any rollout score below this is classified
  as a death (death penalties range from -175 to -500; worst non-death step is
  rotten fruit at -20)
- `RISK_LAMBDA = 2.0` — how hard to penalise lethal paths
- `death_rate` = fraction of the `n_samples` rollouts for this action that
  ended in death
- Result: consistent safe paths beat high-variance paths that occasionally
  score big but sometimes kill the snake

**Previous algorithm (now replaced):** used `max(samples)` — optimistic
pick-the-luckiest-run.  Problem: a risky path that dies 40% of the time but
sometimes returns +100 would beat a safe path averaging +40.

---

## Key files

| File | Role |
|------|------|
| `agent/lookahead.py` | All lookahead logic: rollout, scoring, parallelism |
| `visualization/dashboard.py` | Calls `lookahead_action_with_scores`; compares agent raw score vs lookahead score; tracks per-step danger predictions |
| `training/eval.py` | Standalone eval script; calls `lookahead_action` |
| `training/train_dqn.py` | `evaluate()` calls `lookahead_action` when `lookahead_depth > 0` |
| `configs/training.yaml` | `training.lookahead_depth` (0 = off) |
| `env/snake_env.py` | `SnakePlusEnv` — Gymnasium env with deepcopy-safe state |
| `agent/dqn_agent.py` | `DQNAgent.select_action(obs, training=False)` used inside rollouts |

---

## Environment facts relevant to the algorithm

- **Grid**: 15×15, deterministic physics (wall/body/obstacle collision)
- **Stochasticity**: object spawning (apples, golden, poison, sour, rotten)
  uses Python's `random` module (global state).  `copy.deepcopy(env)` copies
  the env's numpy RNG (`np_random`) but NOT Python's global `random` state,
  so successive deepcopies from the same base state produce different spawn
  sequences.
- **Rewards**: apple +10, golden +30–70, sour −5, rotten −20, poison death −500,
  wall/body/obstacle death −175 to −500 (scales down with snake length),
  step −0.1
- **Observation types**: `"features"` (29-dim vector, MLP agent) or `"grid"`
  (6-channel tensor, CNN agent).  The loaded checkpoint determines which.
- **Actions**: 0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT (relative to current
  direction — there is no 180° reversal)
- **Termination**: wall collision, body collision, obstacle collision, poison
  eaten, starvation (400 steps without eating), or snake length < 1

---

## Known weaknesses of the current algorithm

1. **Greedy rollout bias**: the simulation uses the same greedy policy, so it
   can't discover paths the agent hasn't learned.  A bad policy leads to bad
   rollouts even when a good path exists.

2. **Fixed depth**: `min(snake_length, 15)` is a heuristic.  Short snakes get
   shallow lookahead (depth 3) even when the danger is further away.

3. **No tree search**: only the first action is chosen by lookahead; the rest
   of the rollout follows greedy policy.  A proper tree search (e.g. beam
   search or MCTS) would explore multiple action sequences.

4. **Risk lambda is fixed**: `RISK_LAMBDA = 2.0` is not tuned.  A higher value
   makes the agent too conservative (avoids all negative-reward objects); a
   lower value allows more gambling.

5. **Sample efficiency**: `n_samples=5` with `depth=15` = 75 env steps + 15
   deepcopies per real step.  This is viable for dashboard/eval but expensive.
   Adaptive depth (deeper only when top-2 actions are close in score) would
   help.

6. **No use of Q-values for lookahead ranking**: the lookahead ignores the
   agent's internal Q-values during aggregation.  A weighted combination of
   rollout return and Q-value could be more stable.

---

## How `dashboard.py` uses the scores

```python
agent_raw = agent.select_action(state, training=False)
la_action, scores = lookahead_action_with_scores(env, agent, state, ...)

# Per-step danger tracking: was agent's prediction in the death range?
if scores[agent_raw] < DEATH_SCORE_THRESHOLD:
    self.deaths_by_action[agent_raw] += 1   # fwd / left / right counter

# Choose whichever scores higher — agent's raw pick or lookahead's pick
return la_action if scores[la_action] > scores[agent_raw] else agent_raw
```

The "Danger preds" panel counter shows how many steps this session the agent's
raw greedy prediction would have walked into death (before lookahead correction).

---

## Run commands

```bash
# Watch agent play with lookahead (depth 15, 5 samples)
uv run python -m visualization.dashboard \
    --model results/runs/20260410_101650/model_ep18000.pt \
    --config results/runs/20260410_101650/config.yaml \
    --lookahead 15 --samples 5

# Eval 20 episodes with lookahead, print per-episode stats
uv run python -m training.eval \
    --config configs/training.yaml \
    --lookahead 15 --episodes 20

# Eval without lookahead (pure greedy baseline)
uv run python -m training.eval --config configs/training.yaml
```

---

## Suggested directions for improvement

- **Beam search**: keep the top-K action sequences at each step rather than
  greedy rollout after the first action
- **Adaptive sampling**: start with 1 sample, add more only when the top-2
  actions are within a margin ε of each other
- **Dynamic depth**: use the agent's Q-value uncertainty (if available) to
  decide how deep to look
- **CVaR instead of mean - λ*death_rate**: Conditional Value at Risk averages
  the worst-α fraction of samples — more principled risk metric
- **UCB-style exploration within lookahead**: add a small exploration bonus to
  under-sampled actions to avoid early commitment
