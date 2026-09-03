# FrameworkTasks

Framework code and concrete test tasks are intentionally separate.

```text
FrameworkTasks/
├── qwen35_merger/
│   ├── alignment/   # preserved earlier new-architecture attempts/results
│   ├── packing/     # preserved earlier 256/512 investigations/results
│   └── feedback/    # final 20260904 feedback tasks; one directory per data/attention semantics
├── templates/qwen35_merger/  # sanitized runnable copies, no private paths/payloads
└── template_validation/      # independent copied-template validation; generated payloads ignored
```

Do not delete or repurpose a prior task because a later one supersedes it. A
Task directory owns model/data/objective semantics; `runs/*.yaml` are execution
variants with separate output roots. Read Framework's
`docs/usage/task-organization.md` and `docs/usage/real-vlm-feedback.md` before
copying a template.
