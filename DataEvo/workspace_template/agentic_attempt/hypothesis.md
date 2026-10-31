# Terminal trajectory baseline

Generate multiple teacher trajectories in independent Harbor environments and keep
verifier-passing, unique trajectories. Compare SFT and RL dev scores, command failure
patterns, turn counts and sandbox time before changing prompts or filtering.

The train task allowlist, environment, verifier and training settings are fixed.
Do not generate answers for held-out tasks or claim infrastructure failures as zero rewards.
