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

    def embed(self, input_ids):
        return self.embedding(input_ids)

    def forward_embeddings(self, embeddings, **kwargs):
        del kwargs
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
    )
    assert result.logits.shape == (1, 3, 7)


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
