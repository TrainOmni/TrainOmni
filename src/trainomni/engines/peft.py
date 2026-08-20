"""Optional PEFT application kept outside model-family plugins."""

from __future__ import annotations

from typing import Any

from trainomni.config import StageSpec


class PeftError(RuntimeError):
    pass


def apply_peft_if_requested(
    model: Any,
    stage: StageSpec,
) -> Any:
    requests = [
        policy.peft
        for policy in stage.component_policy.values()
        if policy.trainable and policy.peft is not None
    ]
    if not requests:
        return model
    first = requests[0]
    if any(
        item.method != first.method
        or item.rank != first.rank
        or item.alpha != first.alpha
        or item.dropout != first.dropout
        or item.task_type != first.task_type
        or item.config != first.config
        for item in requests[1:]
    ):
        raise PeftError(
            "one stage currently requires a shared LoRA/QLoRA hyperparameter set"
        )
    targets = tuple(dict.fromkeys(
        target for item in requests for target in item.target_modules
    ))
    modules_to_save = tuple(dict.fromkeys(
        target for item in requests for target in item.modules_to_save
    ))
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PeftError(
            "PEFT was requested; install trainomni-framework[peft]"
        ) from exc
    if first.method == "qlora":
        quantized = bool(
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
        )
        if not quantized:
            raise PeftError(
                "QLoRA requires the model plugin to load a 4-bit or 8-bit model"
            )
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=first.rank,
        lora_alpha=first.alpha,
        lora_dropout=first.dropout,
        target_modules=list(targets),
        modules_to_save=list(modules_to_save) or None,
        task_type=first.task_type,
        **dict(first.config),
    )
    return get_peft_model(model, lora_config)
