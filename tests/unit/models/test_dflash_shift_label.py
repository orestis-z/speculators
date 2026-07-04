"""Tests for DFlash shift_label config and speculative_tokens logic."""

from unittest.mock import patch

import pytest
from transformers import Qwen3Config

from speculators import VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel


@pytest.fixture
def tiny_verifier_config():
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
    )


_FAKE_VERIFIER = VerifierConfig(name_or_path="dummy", architectures=[])


class TestShiftLabelConfig:
    def test_default_shift_label_is_false(self):
        config = DFlashSpeculatorConfig()
        assert config.shift_label is False

    def test_shift_label_stored_in_config(self):
        config = DFlashSpeculatorConfig(shift_label=True)
        assert config.shift_label is True


class TestShiftLabelSpeculativeTokens:
    @staticmethod
    def _build_kwargs(tiny_verifier_config, **overrides):
        base = {
            "draft_vocab_size": 32,
            "block_size": 8,
            "verifier_name_or_path": "dummy",
            "target_layer_ids": [0],
        }
        base.update(overrides)
        with patch.object(
            VerifierConfig, "from_pretrained", return_value=_FAKE_VERIFIER
        ):
            return DFlashDraftModel._build_base_config_kwargs(
                "dflash", tiny_verifier_config, **base
            )

    def test_no_shift_gives_block_size_minus_one(self, tiny_verifier_config):
        result = self._build_kwargs(tiny_verifier_config, shift_label=False)
        spec_tokens = result["speculators_config"].proposal_methods[
            0
        ].speculative_tokens
        assert spec_tokens == 7

    def test_shift_gives_block_size(self, tiny_verifier_config):
        result = self._build_kwargs(tiny_verifier_config, shift_label=True)
        spec_tokens = result["speculators_config"].proposal_methods[
            0
        ].speculative_tokens
        assert spec_tokens == 8

    def test_default_no_shift(self, tiny_verifier_config):
        result = self._build_kwargs(tiny_verifier_config)
        spec_tokens = result["speculators_config"].proposal_methods[
            0
        ].speculative_tokens
        assert spec_tokens == 7

    @pytest.mark.parametrize("block_size", [4, 8, 16])
    def test_various_block_sizes(self, tiny_verifier_config, block_size):
        no_shift = self._build_kwargs(
            tiny_verifier_config, block_size=block_size, shift_label=False
        )
        with_shift = self._build_kwargs(
            tiny_verifier_config, block_size=block_size, shift_label=True
        )
        assert (
            no_shift["speculators_config"]
            .proposal_methods[0]
            .speculative_tokens
            == block_size - 1
        )
        assert (
            with_shift["speculators_config"]
            .proposal_methods[0]
            .speculative_tokens
            == block_size
        )

    def test_shift_label_propagated_to_config_dict(self, tiny_verifier_config):
        result_on = self._build_kwargs(
            tiny_verifier_config, shift_label=True
        )
        assert result_on["shift_label"] is True

        result_off = self._build_kwargs(
            tiny_verifier_config, shift_label=False
        )
        assert result_off["shift_label"] is False
