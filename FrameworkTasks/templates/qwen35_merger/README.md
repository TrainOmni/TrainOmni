# Qwen3.5 raw-ViT + random merger + MiniCPM5 templates

These are sanitized copies of real tasks validated on 2026-09-04. Pick exactly
one variant, copy that whole directory to a new `YYYYMMDD_specific_task`, copy
`paths.example.json` to `paths.local.json`, edit only the three local asset/data
paths, and follow Framework's `docs/usage/real-vlm-feedback.md`.

The four variants are independent: `unpacked`, dense `packed` Parquet, dense
`arrow` packing and explicit `varlen`. They include source and commands, but no
private paths, generated data, task binding, output, logs, weights or checkpoint.
Do not run in this template directory and then overwrite it for another experiment.
