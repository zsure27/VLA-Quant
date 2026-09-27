# VLA-Quant experiment admission

Before proposing, coding, launching, or extending any GPU experiment, execute the full [Research Governor](docs/VLA_RESEARCH_GOVERNOR_20260926.txt). This is a required admission gate, including for a resumed or automated stage.

Read the current master plan, experiment log, latest accepted session report, canonical baseline table, and stage gate decision. Record the current primary stage, locked stages, and one main research question. Resolve conflicting results by source audit before using them as a baseline.

Save a pre-run card with hypothesis, one changed variable, fixed control, primary metric, positive and negative decisions, data split, backbone hash, output directory, cost estimate, and stopping condition. Classify each run as contract, diagnostic, development, or final evaluation. Do not run an experiment if either outcome would merely trigger another similar variant.

After each run, save a post-run decision stating what is supported, what remains confounded, whether the stage or branch advances, and the single next question. Apply the project's quota, backup, and shutdown rules independently of scientific admission. When no admissible experiment remains on an active paid instance, archive and close it rather than waiting for user input.

Current primary stage: P2.5 closed-loop alignment diagnosis. Exact 12L backbone, blocks 18–19, rank8 Recovery-LoRA. P4 experts and P5 Router are locked. The 50-episode pilot has no observed 12L-to-14L headroom and cannot yield a W4 recovery fraction. See [P2.5 plan](docs/P2_5_ON_POLICY_ALIGNMENT_PLAN_20260926_CN.md) for the current evidence and allowed controls. New canonical evidence may change this state only through a versioned gate decision.
