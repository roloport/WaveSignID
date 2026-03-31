"""Unit tests for the WaveID_Agent class."""

import numpy as np
import pytest

from waveid.identity import WaveID_Agent
from waveid.protocol import MASK_SIZE, Challenge, generate_nonce
from waveid.crypto import generate_challenge_matrix

N = 64


class TestAgentCreation:
    def test_creates_with_id(self):
        agent = WaveID_Agent("agent-001", mask_size=N, seed=1)
        assert agent.agent_id == "agent-001"
        assert agent.mask_size == N

    def test_public_key_shape(self):
        agent = WaveID_Agent("agent-001", mask_size=N, seed=1)
        pk = agent.get_public_key()
        assert pk.shape == (N, N)
        assert pk.dtype == np.float64

    def test_verify_key_bytes(self):
        agent = WaveID_Agent("agent-001", mask_size=N, seed=1)
        vk = agent.get_verify_key()
        assert isinstance(vk, bytes)
        assert len(vk) == 32  # Ed25519 public key is 32 bytes

    def test_deterministic_with_seed(self):
        a1 = WaveID_Agent("a", mask_size=N, seed=42)
        a2 = WaveID_Agent("a", mask_size=N, seed=42)
        np.testing.assert_array_equal(a1.get_public_key(), a2.get_public_key())
        assert a1.get_verify_key() == a2.get_verify_key()


class TestSignChallenge:
    def test_produces_valid_response(self):
        agent = WaveID_Agent("agent-001", mask_size=N, seed=1)
        challenge = Challenge(
            agent_id="agent-001",
            nonce=generate_nonce(),
            matrix=generate_challenge_matrix(N),
            timestamp=0.0,
        )
        response = agent.sign_challenge(challenge)
        assert response.agent_id == "agent-001"
        assert response.nonce == challenge.nonce
        assert response.wave_proof.shape == (N, N)
        assert isinstance(response.signature, bytes)
        assert len(response.signature) == 64  # Ed25519 signature

    def test_different_challenges_different_proofs(self):
        agent = WaveID_Agent("agent-001", mask_size=N, seed=1)
        c1 = Challenge("agent-001", generate_nonce(), generate_challenge_matrix(N), 0.0)
        c2 = Challenge("agent-001", generate_nonce(), generate_challenge_matrix(N), 0.0)
        r1 = agent.sign_challenge(c1)
        r2 = agent.sign_challenge(c2)
        assert not np.allclose(r1.wave_proof, r2.wave_proof)


class TestExportImport:
    def test_roundtrip(self):
        original = WaveID_Agent("agent-001", mask_size=N, seed=1)
        data = original.export_identity()
        restored = WaveID_Agent.from_identity(data)

        assert restored.agent_id == original.agent_id
        assert restored.mask_size == original.mask_size
        np.testing.assert_allclose(
            restored.get_public_key(), original.get_public_key(), atol=1e-10
        )
        assert restored.get_verify_key() == original.get_verify_key()

    def test_restored_agent_can_sign(self):
        original = WaveID_Agent("agent-001", mask_size=N, seed=1)
        data = original.export_identity()
        restored = WaveID_Agent.from_identity(data)

        challenge = Challenge(
            agent_id="agent-001",
            nonce=generate_nonce(),
            matrix=generate_challenge_matrix(N),
            timestamp=0.0,
        )
        response = restored.sign_challenge(challenge)
        assert response.wave_proof.shape == (N, N)
        assert len(response.signature) == 64
