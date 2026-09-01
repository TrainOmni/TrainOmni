from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trainomni.contracts.features import (
    ModalFeatureBranch,
    ModalFeatures,
    ModalFeatureSet,
)
from trainomni.core.errors import SpecError
from trainomni.modules.model.connectors.linear.config import LinearConnectorConfig
from trainomni.modules.model.connectors.linear.module import LinearConnector
from trainomni.modules.model.fusions.cross_attention.config import (
    CrossAttentionFusionConfig,
)
from trainomni.modules.model.fusions.cross_attention.module import CrossAttentionFusion
from trainomni.modules.model.fusions.prefix.config import PrefixFusionConfig
from trainomni.modules.model.fusions.prefix.module import PrefixFusion
from trainomni.modules.model.fusions.token_replace.config import TokenReplaceConfig
from trainomni.modules.model.fusions.token_replace.module import TokenReplaceFusion


class Language(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.head = nn.Linear(4, 7)
        self.received = None

    def embed(self, input_ids):
        return self.embedding(input_ids)

    def forward_embeddings(self, embeddings, **kwargs):
        self.received = {"embeddings": embeddings, **kwargs}
        return SimpleNamespace(logits=self.head(embeddings))


def test_linear_connector_preserves_feature_structure() -> None:
    connector = LinearConnector(
        LinearConnectorConfig(input_dim=3, output_dim=4, bias=False)
    )
    features = ModalFeatures(
        embeddings=torch.ones(2, 5, 3),
        mask=torch.ones(2, 5, dtype=torch.bool),
        grid=(1, 5),
        metadata={"modality": "image"},
    )
    result = connector(features)
    assert result.embeddings.shape == (2, 5, 4)
    assert result.mask is features.mask
    assert result.grid == (1, 5)
    assert result.metadata["modality"] == "image"


def test_prefix_fusion_returns_text_aligned_logits() -> None:
    language = Language()
    input_ids = torch.tensor([[1, 2, 3]])
    result = PrefixFusion(PrefixFusionConfig())(
        language=language,
        input_ids=input_ids,
        modal_features=ModalFeatures(torch.randn(1, 2, 4)),
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(3).unsqueeze(0),
    )
    assert result.logits.shape == (1, 3, 7)
    assert language.received["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert language.received["position_ids"].dtype == torch.int64
    assert language.received["attention_mask"].shape == (1, 5)
    for field in ("cache_position", "rope_deltas"):
        with pytest.raises(SpecError, match="cannot generically rewrite"):
            PrefixFusion(PrefixFusionConfig())(
                language=language,
                input_ids=input_ids,
                modal_features=ModalFeatures(torch.randn(1, 2, 4)),
                attention_mask=torch.ones_like(input_ids),
                **{field: torch.arange(3)},
            )


def test_prefix_fusion_rebases_positions_over_only_valid_modal_tokens() -> None:
    language = Language()
    input_ids = torch.tensor([[1, 2, 0]])
    PrefixFusion(PrefixFusionConfig())(
        language=language,
        input_ids=input_ids,
        modal_features=ModalFeatures(
            torch.randn(1, 2, 4),
            mask=torch.tensor([[True, False]]),
        ),
        attention_mask=torch.tensor([[1, 1, 0]]),
        position_ids=torch.tensor([[0, 1, 0]], dtype=torch.int32),
    )
    assert language.received["attention_mask"].tolist() == [[1, 0, 1, 1, 0]]
    assert language.received["position_ids"].tolist() == [[0, 0, 1, 2, 0]]
    assert language.received["position_ids"].dtype == torch.int32

    with pytest.raises(SpecError, match="must align"):
        PrefixFusion(PrefixFusionConfig())(
            language=language,
            input_ids=input_ids,
            modal_features=ModalFeatures(torch.randn(1, 2, 4)),
            position_ids=torch.arange(4).unsqueeze(0),
        )


def test_token_replacement_validates_positions() -> None:
    language = Language()
    fusion = TokenReplaceFusion(TokenReplaceConfig())
    input_ids = torch.tensor([[1, 2, 3]])
    modal = ModalFeatures(torch.randn(1, 2, 4))
    result = fusion(
        language=language,
        input_ids=input_ids,
        modal_features=modal,
        modal_positions=torch.tensor([[0, 2]]),
    )
    assert result.logits.shape == (1, 3, 7)
    with pytest.raises(SpecError, match="duplicate"):
        fusion(
            language=language,
            input_ids=input_ids,
            modal_features=modal,
            modal_positions=torch.tensor([[1, 1]]),
        )


def test_token_replacement_supports_unequal_modal_counts_with_masked_padding() -> None:
    language = Language()
    fusion = TokenReplaceFusion(TokenReplaceConfig())
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    modal = torch.randn(2, 2, 4)
    mask = torch.tensor([[True, False], [True, True]])
    result = fusion(
        language=language,
        input_ids=input_ids,
        modal_features=ModalFeatures(modal, mask=mask),
        modal_positions=torch.tensor([[1, -1], [0, 2]]),
    )
    assert result.logits.shape == (2, 3, 7)
    expected = language.embed(input_ids).detach()
    received = language.received["embeddings"].detach()
    torch.testing.assert_close(received[0, 0], expected[0, 0])
    torch.testing.assert_close(received[0, 2], expected[0, 2])
    torch.testing.assert_close(received[0, 1], modal[0, 0])
    with pytest.raises(SpecError, match="padded modal slots"):
        fusion(
            language=language,
            input_ids=input_ids,
            modal_features=ModalFeatures(modal, mask=mask),
            modal_positions=torch.tensor([[1, 0], [0, 2]]),
        )


def test_cross_attention_fusion_routes_memory_and_mask() -> None:
    class CrossAttentionLanguage(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.received = None

        def embed(self, input_ids):
            return self.embedding(input_ids)

        def forward_embeddings(self, embeddings, **kwargs):
            self.received = kwargs
            return {"logits": embeddings}

    language = CrossAttentionLanguage()
    features = ModalFeatureSet(
        (
            ModalFeatureBranch(
                "vision",
                "image",
                ModalFeatures(
                    torch.ones(1, 2, 4),
                    mask=torch.tensor([[True, False]]),
                ),
            ),
        )
    )
    output = CrossAttentionFusion(CrossAttentionFusionConfig())(
        language=language,
        input_ids=torch.tensor([[1, 2, 3]]),
        modal_features=features,
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        modal_positions=None,
    )
    assert output["logits"].shape == (1, 3, 4)
    assert tuple(language.received["encoder_hidden_states"].shape) == (1, 2, 4)
    assert torch.equal(
        language.received["encoder_attention_mask"],
        torch.tensor([[True, False]]),
    )
