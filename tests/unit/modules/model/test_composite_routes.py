from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from trainomni.contracts.features import (
    ModalFeatureBranch,
    ModalFeatures,
    ModalFeatureSet,
)
from trainomni.core.context import BuildContext
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleId, ModuleRef
from trainomni.modules.model.models.composite.config import (
    CompositeBranchConfig,
)
from trainomni.modules.model.models.composite.module import CompositeModel, descriptor


class ScaleEncoder(nn.Module):
    def __init__(self, initial: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial))

    def forward(self, values: torch.Tensor) -> ModalFeatures:
        return ModalFeatures(values * self.scale)


class ScaleConnector(nn.Module):
    def __init__(self, initial: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial))

    def forward(self, features: ModalFeatures) -> ModalFeatures:
        return ModalFeatures(
            embeddings=features.embeddings * self.scale,
            mask=features.mask,
            positions=features.positions,
            grid=features.grid,
            metadata=features.metadata,
        )


class TinyLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(9, 4)
        self.head = nn.Linear(4, 9)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward_embeddings(self, embeddings: torch.Tensor, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.head(embeddings))


class RecordingFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received: ModalFeatureSet | None = None

    def forward(
        self,
        *,
        language,
        input_ids,
        modal_features,
        attention_mask=None,
        modal_positions=None,
        **kwargs,
    ):
        del attention_mask, kwargs
        assert isinstance(modal_features, ModalFeatureSet)
        self.received = modal_features
        merged = modal_features.concatenate()
        assert torch.equal(merged.positions, modal_positions)
        signal = merged.embeddings.mean(dim=1, keepdim=True)
        return language.forward_embeddings(language.embed(input_ids) + signal)


def branches() -> tuple[CompositeBranchConfig, ...]:
    return (
        CompositeBranchConfig(
            name="vision",
            modality="image",
            input_key="pixel_values",
            positions_key="image_positions",
            encoder="vision_encoder",
            connector="vision_connector",
        ),
        CompositeBranchConfig(
            name="motion",
            modality="video",
            input_key="video_values",
            positions_key="video_positions",
            encoder="video_encoder",
            connector="video_connector",
            required=False,
        ),
    )


def components() -> dict[str, nn.Module]:
    return {
        "vision_encoder": ScaleEncoder(1.0),
        "vision_connector": ScaleConnector(1.0),
        "video_encoder": ScaleEncoder(1.0),
        "video_connector": ScaleConnector(1.0),
        "fusion_core": RecordingFusion(),
        "decoder": TinyLanguage(),
    }


def test_composite_routes_ordered_branches_and_gradients() -> None:
    model = CompositeModel(
        branches=branches(),
        components=components(),
        fusion_name="fusion_core",
        language_name="decoder",
    )
    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        pixel_values=torch.ones(1, 1, 4),
        image_positions=torch.tensor([[0]]),
        video_values=torch.full((1, 2, 4), 2.0),
        video_positions=torch.tensor([[1, 2]]),
    )
    output.logits.sum().backward()

    fusion = model.get_submodule("fusion_core")
    assert isinstance(fusion, RecordingFusion)
    assert fusion.received is not None
    assert tuple(branch.name for branch in fusion.received.branches) == (
        "vision",
        "motion",
    )
    merged = fusion.received.concatenate()
    assert merged.metadata["branch_slices"] == {
        "vision": (0, 1),
        "motion": (1, 3),
    }
    assert torch.equal(merged.positions, torch.tensor([[0, 1, 2]]))
    for path in (
        "vision_encoder.scale",
        "vision_connector.scale",
        "video_encoder.scale",
        "video_connector.scale",
    ):
        parameter = dict(model.named_parameters())[path]
        assert parameter.grad is not None
        assert bool(parameter.grad.ne(0).any().item())


def test_optional_branch_can_be_absent_but_required_branch_fails_closed() -> None:
    model = CompositeModel(
        branches=branches(),
        components=components(),
        fusion_name="fusion_core",
        language_name="decoder",
    )
    model(
        input_ids=torch.tensor([[1, 2]]),
        pixel_values=torch.ones(1, 1, 4),
        image_positions=torch.tensor([[0]]),
    )
    fusion = model.get_submodule("fusion_core")
    assert isinstance(fusion, RecordingFusion)
    assert fusion.received is not None
    assert tuple(branch.name for branch in fusion.received.branches) == ("vision",)

    with pytest.raises(SpecError, match="required branch 'vision'"):
        model(
            input_ids=torch.tensor([[1, 2]]),
            video_values=torch.ones(1, 1, 4),
            video_positions=torch.tensor([[0]]),
        )


def test_branch_position_disagreement_and_partial_positions_fail_closed() -> None:
    class PositionedEncoder(ScaleEncoder):
        def forward(self, values: torch.Tensor) -> ModalFeatures:
            return ModalFeatures(
                values * self.scale,
                positions=torch.tensor([[1]], device=values.device),
            )

    owned = components()
    owned["vision_encoder"] = PositionedEncoder(1.0)
    model = CompositeModel(
        branches=branches(),
        components=owned,
        fusion_name="fusion_core",
        language_name="decoder",
    )
    with pytest.raises(SpecError, match="positions disagree"):
        model(
            input_ids=torch.tensor([[1, 2]]),
            pixel_values=torch.ones(1, 1, 4),
            image_positions=torch.tensor([[0]]),
        )

    feature_set = ModalFeatureSet.coerce(ModalFeatures(torch.ones(1, 1, 4)))
    assert feature_set.concatenate().embeddings.shape == (1, 1, 4)
    partial_positions = ModalFeatureSet(
        (
            ModalFeatureBranch(
                "image",
                "image",
                ModalFeatures(
                    torch.ones(1, 1, 4), positions=torch.tensor([[0]])
                ),
            ),
            ModalFeatureBranch(
                "video", "video", ModalFeatures(torch.ones(1, 1, 4))
            ),
        )
    )
    with pytest.raises(ValueError, match="every branch or no branch"):
        partial_positions.concatenate()


def test_nested_branch_configuration_and_custom_component_names_build() -> None:
    reference = ModuleRef(
        ModuleId.parse("model:trainomni/composite@1"),
        {
            "branches": (
                {
                    "name": "vision",
                    "modality": "image",
                    "input_key": "pixel_values",
                    "encoder": "vision_encoder",
                    "connector": "vision_connector",
                    "positions_key": "image_positions",
                },
            ),
            "fusion": "fusion_core",
            "language": "decoder",
        },
    )
    owned = components()
    model = descriptor().build(
        reference,
        BuildContext(
            task_digest="task",
            components=MappingProxyType(owned),
        ),
    )
    assert isinstance(model, CompositeModel)
    assert set(dict(model.named_children())) == {
        "vision_encoder",
        "vision_connector",
        "fusion_core",
        "decoder",
    }

    with pytest.raises(SpecError, match="branch names must be unique"):
        descriptor().build(
            ModuleRef(
                reference.module_id,
                {
                    "branches": (
                        dict(reference.config["branches"][0]),
                        dict(reference.config["branches"][0]),
                    ),
                    "fusion": "fusion_core",
                    "language": "decoder",
                },
            ),
            BuildContext(task_digest="task", components=MappingProxyType(owned)),
        )
