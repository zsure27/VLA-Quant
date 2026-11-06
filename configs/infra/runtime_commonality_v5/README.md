# Independent drift study v5

CPU-only specification: three models, four LIBERO suites, ten tasks per suite and five initial states. Twelve static conditions including BF16; QVLA and DyQ excluded. No GPU authorization or method compatibility is implied.

`python -m qvla.runtime_study_v5 --gpu-slots 4 --output .local/drift-plan-NEW` generates sampling/shard proposals only. Actual GPU count is unknown; pass a slot count only for planning.

See [design](../../../Ex_Reports/_tasks/multisuite-trajectory-deviation-v1/DESIGN_V5_CN.md).
