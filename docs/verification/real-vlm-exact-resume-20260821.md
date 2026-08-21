# Real VLM five-route exact-resume validation — 2026-08-21

This gate validates checkpoint continuity, not model quality. The real Qwen3.5
vision + MiniCPM5-1B composite ran on one RTX 4060 Ti with true BF16. Each route
used one immutable Task/Run identity and the following protocol:

```text
uninterrupted steps 1→4
compare against
steps 1→2 → atomic checkpoint → fresh Python process → resume steps 3→4
```

The comparison hashes every named model tensor and every logical AdamW state
value, and also requires the runtime checkpoint digest, scheduler/objective/data
state, final loss terms, metrics and parameter-update evidence to match exactly.
PyTorch/safetensors container-file bytes are not used as the equality oracle:
equivalent logical state can have different serialization container bytes.

| Route | Restored state | Final train evidence | Held-out evaluation |
| --- | --- | --- | --- |
| Full-parameter SFT | model `eb83c7dd…`; optimizer `fe023bfc…`; runtime `c202c7bc…` | token CE `21.38323`; vision/connector/LLM update evidence identical | `sft_loss=7.28938` |
| Multimodal pretraining | model `b9ea0f4a…`; optimizer `f9ce5956…`; runtime `17a62686…` | token CE `12.62012`; both component activation-checkpoint hooks active | `pretraining_loss=6.37378` |
| Connector alignment | model `6a68298d…`; optimizer `ab50cf93…`; runtime `6fa1b117…` | token CE `3.32854`; only connector optimizer group restored and updated | `alignment_loss=2.81367` |
| Offline dense-logit KD | model `75b6b77f…`; optimizer `a4b1da1c…`; runtime `85656f1a…` | CE `5.52228`, dense KL `0.17443`; cache/data/objective cursor restored | `kd_ce=5.05566`, `kd_kl=0.12258` |
| Offline-reference DPO | model `9aa4705a…`; optimizer `bf07ecf8…`; runtime `8da57811…` | DPO `0.02983`; both branches, cached reference data and objective counters restored | `dpo_loss=0.00406` |

All five receipts report:

- uninterrupted step 4 and resumed step 4 logical checkpoint state equal;
- final loss/gradient/learning-rate/objective/data/update evidence equal;
- resume performed in a fresh Python process;
- the resumed checkpoint remains loadable by held-out evaluation.

Durable receipts and the reusable verifier live outside Framework under
`D:\Codex\TrainOmni\FrameworkValidation\resume-validation` and
`D:\Codex\TrainOmni\FrameworkValidation\verify_real_exact_resume.py`.
Large reference and resumed checkpoint payloads were created under the system
temporary directory and removed after their logical digests and evaluation
evidence were recorded.
