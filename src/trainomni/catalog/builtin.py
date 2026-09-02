"""Explicit builtin descriptors.

Importing this module does not create a registry or construct a model. Callers
choose whether to include the builtin catalog in their own registry instance.
"""

from trainomni.core.registry import ModuleRegistry
from trainomni.modules.data.adapters.msswift.module import (
    descriptor as msswift_adapter,
)
from trainomni.modules.data.collation.multimodal.module import (
    descriptor as multimodal_collator,
)
from trainomni.modules.data.collation.padding_free.module import (
    descriptor as padding_free_collator,
)
from trainomni.modules.data.model_io.transformers.module import (
    descriptor as transformers_model_io,
)
from trainomni.modules.data.packing.none.module import descriptor as no_packing
from trainomni.modules.data.packing.padding_free.module import descriptor as padding_free_packing
from trainomni.modules.data.packing.sequence.module import descriptor as sequence_packing
from trainomni.modules.data.sources.arrow.module import descriptor as arrow_source
from trainomni.modules.data.sources.jsonl.module import descriptor as jsonl_source
from trainomni.modules.data.sources.memory.module import descriptor as memory_source
from trainomni.modules.data.sources.mixture.module import descriptor as mixture_source
from trainomni.modules.data.sources.parquet.module import descriptor as parquet_source
from trainomni.modules.data.supervision.causal_lm.module import (
    descriptor as causal_supervision,
)
from trainomni.modules.data.supervision.dense_kd.module import (
    descriptor as dense_kd_supervision,
)
from trainomni.modules.data.supervision.preference.module import (
    descriptor as preference_supervision,
)
from trainomni.modules.data.transforms.image.module import descriptor as image_transform
from trainomni.modules.data.transforms.media.module import descriptor as media_transform
from trainomni.modules.data.transforms.tensor_cache.module import (
    descriptor as tensor_cache_transform,
)
from trainomni.modules.data.transforms.video.module import descriptor as video_transform
from trainomni.modules.evaluation.loss.module import descriptor as loss_evaluator
from trainomni.modules.evaluation.task_metrics.module import (
    descriptor as task_metrics_evaluator,
)
from trainomni.modules.export.lora_adapter.module import (
    descriptor as lora_adapter_exporter,
)
from trainomni.modules.export.safetensors.module import descriptor as safetensors_exporter
from trainomni.modules.export.transformers.module import (
    descriptor as transformers_exporter,
)
from trainomni.modules.model.attention.packed.module import (
    descriptor as packed_attention,
)
from trainomni.modules.model.attention.policies.module import (
    descriptor as model_default_attention,
)
from trainomni.modules.model.connectors.linear.module import descriptor as linear_connector
from trainomni.modules.model.connectors.mlp.module import descriptor as mlp_connector
from trainomni.modules.model.encoders.transformers_video.module import (
    descriptor as transformers_video,
)
from trainomni.modules.model.encoders.transformers_vision.module import (
    descriptor as transformers_vision,
)
from trainomni.modules.model.fusions.cross_attention.module import (
    descriptor as cross_attention_fusion,
)
from trainomni.modules.model.fusions.prefix.module import descriptor as prefix_fusion
from trainomni.modules.model.fusions.token_replace.module import (
    descriptor as token_replace_fusion,
)
from trainomni.modules.model.language.transformers_causal_lm.module import (
    descriptor as transformers_causal_language,
)
from trainomni.modules.model.models.composite.module import descriptor as composite_model
from trainomni.modules.model.models.monolithic.module import (
    descriptor as monolithic_transformers,
)
from trainomni.modules.objectives.causal_lm.module import descriptor as causal_lm
from trainomni.modules.objectives.dense_kd.module import descriptor as dense_kd
from trainomni.modules.objectives.dpo.module import descriptor as dpo
from trainomni.modules.parameters.component.module import (
    descriptor as component_parameters,
)
from trainomni.modules.parameters.freeze.module import descriptor as freeze_parameters
from trainomni.modules.parameters.full.module import descriptor as full_parameters
from trainomni.modules.parameters.lora.module import descriptor as lora_parameters


def builtin_descriptors():
    return (
        memory_source(),
        jsonl_source(),
        parquet_source(),
        arrow_source(),
        mixture_source(),
        msswift_adapter(),
        media_transform(),
        tensor_cache_transform(),
        image_transform(),
        video_transform(),
        transformers_model_io(),
        causal_supervision(),
        dense_kd_supervision(),
        preference_supervision(),
        no_packing(),
        sequence_packing(),
        padding_free_packing(),
        multimodal_collator(),
        padding_free_collator(),
        transformers_vision(),
        transformers_video(),
        model_default_attention(),
        packed_attention(),
        linear_connector(),
        mlp_connector(),
        prefix_fusion(),
        cross_attention_fusion(),
        token_replace_fusion(),
        transformers_causal_language(),
        composite_model(),
        monolithic_transformers(),
        causal_lm(),
        dense_kd(),
        dpo(),
        loss_evaluator(),
        task_metrics_evaluator(),
        safetensors_exporter(),
        lora_adapter_exporter(),
        transformers_exporter(),
        full_parameters(),
        component_parameters(),
        freeze_parameters(),
        lora_parameters(),
    )


def builtin_registry() -> ModuleRegistry:
    return ModuleRegistry(builtin_descriptors())
