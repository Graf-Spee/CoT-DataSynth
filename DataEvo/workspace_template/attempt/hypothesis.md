# Baseline: verified teacher solutions

Evidence: no completed attempts yet. Generate two teacher solutions per training
task, retain only solutions matching the known GSM8K answer, remove exact duplicate
solutions. The expected benefit is a reliable cold-start baseline. Possible
regression: narrow coverage and overly easy tasks. Compare fixed SFT/RL dev scores,
probe failures, length distribution and filtering yield before the next attempt.
