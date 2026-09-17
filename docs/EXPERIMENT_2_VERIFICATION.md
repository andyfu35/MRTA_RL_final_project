# Experiment 2 Verification Notes

Verified in the implementation sandbox on 2026-09-17:

- focused worker/trainer/evaluation/config tests: 12 passed
- Python compilation succeeded for `two_runner.py`, `two_runner_env.py`, `policy.py`, `config.py`, and `cli.py`
- CLI parser regression confirmed existing `train`, `render`, `single-train`, `single-eval` commands remain parseable alongside `two-train` and `two-eval`
- real `TwoRunnerArena2D` end-to-end smoke completed one synchronous PPO round from a synthetic 15-D warm-start checkpoint, produced both `latest.pt` and `round_00001.pt`, and each Runner consumed the exact configured valid sample count with zero queued cross-round surplus

Sandbox limitation: the isolated reconstructed workspace did not contain the repository's complete pre-existing test suite, so this is not a claim that the full repository `pytest -q` has been executed in the sandbox. Run the full suite from a complete local checkout before treating the branch as fully regression-verified.
