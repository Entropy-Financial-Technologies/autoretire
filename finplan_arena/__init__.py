"""FinPlan Arena: an evaluation harness for LLM agents on multi-decade
household financial planning.

The simulator is deterministic: given (scenario, decision sequence, return
sequence) the outcome is fully reproducible. All randomness lives in the
seeded return generator, and trial *k* uses the same return path for every
agent (paired-seed design).
"""

__version__ = "0.1.0"
