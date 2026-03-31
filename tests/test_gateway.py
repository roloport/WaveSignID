"""Unit tests for the WaveID_Gateway class."""

import numpy as np
import pytest

from waveid.gateway import WaveID_Gateway
from waveid.identity import WaveID_Agent
from waveid.protocol import AuthResult

N = 64


def make_registered_pair(agent_id: str = "agent-001", seed: int = 1):
    """Create an agent and gateway with the agent registered."""
    agent = WaveID_Agent(agent_id, mask_size=N, seed=seed)
    gateway = WaveID_Gateway(mask_size=N)
    gateway.register(agent_id, agent.get_public_key(), agent.get_verify_key())
    return agent, gateway


class TestRegistration:
    def test_register_agent(self):
        agent = WaveID_Agent("agent-001", mask_size=N, seed=1)
        gateway = WaveID_Gateway(mask_size=N)
        gateway.register("agent-001", agent.get_public_key(), agent.get_verify_key())
        assert "agent-001" in gateway.registered_agents

    def test_issue_challenge_unregistered(self):
        gateway = WaveID_Gateway(mask_size=N)
        with pytest.raises(KeyError):
            gateway.issue_challenge("unknown")


class TestChallengeIssuance:
    def test_challenge_fields(self):
        agent, gateway = make_registered_pair()
        challenge = gateway.issue_challenge("agent-001")
        assert challenge.agent_id == "agent-001"
        assert len(challenge.nonce) == 32
        assert challenge.matrix.shape == (N, N)
        assert challenge.timestamp > 0


class TestVerification:
    def test_valid_authentication(self):
        agent, gateway = make_registered_pair()
        challenge = gateway.issue_challenge("agent-001")
        response = agent.sign_challenge(challenge)
        result = gateway.verify(response)
        assert result.authenticated is True
        assert result.result == AuthResult.SUCCESS
        assert result.correlation_score is not None
        assert result.correlation_score > 0.3

    def test_unknown_agent_rejected(self):
        agent, gateway = make_registered_pair()
        challenge = gateway.issue_challenge("agent-001")
        response = agent.sign_challenge(challenge)
        # Tamper with agent_id
        from waveid.protocol import Response
        tampered = Response(
            agent_id="unknown-agent",
            nonce=response.nonce,
            wave_proof=response.wave_proof,
            signature=response.signature,
        )
        result = gateway.verify(tampered)
        assert result.authenticated is False
        assert result.result == AuthResult.UNKNOWN_AGENT


class TestForgeryDetection:
    def test_wrong_mask_rejected(self):
        """An attacker with a different phase mask should be rejected."""
        real_agent, gateway = make_registered_pair(seed=1)
        fake_agent = WaveID_Agent("agent-001", mask_size=N, seed=999)

        challenge = gateway.issue_challenge("agent-001")
        # Fake agent tries to sign but has wrong mask AND wrong Ed25519 key
        fake_response = fake_agent.sign_challenge(challenge)
        result = gateway.verify(fake_response)
        assert result.authenticated is False
        # Should fail at signature check since Ed25519 key doesn't match
        assert result.result == AuthResult.INVALID_SIGNATURE


class TestReplayPrevention:
    def test_nonce_reuse_rejected(self):
        agent, gateway = make_registered_pair()
        challenge = gateway.issue_challenge("agent-001")
        response = agent.sign_challenge(challenge)

        # First use succeeds
        result1 = gateway.verify(response)
        assert result1.authenticated is True

        # Replay attempt fails (nonce already consumed)
        result2 = gateway.verify(response)
        assert result2.authenticated is False
        assert result2.result == AuthResult.INVALID_NONCE

    def test_fabricated_nonce_rejected(self):
        agent, gateway = make_registered_pair()
        challenge = gateway.issue_challenge("agent-001")
        response = agent.sign_challenge(challenge)

        # Tamper with nonce
        from waveid.protocol import Response
        tampered = Response(
            agent_id=response.agent_id,
            nonce=b"\x00" * 32,  # fabricated nonce
            wave_proof=response.wave_proof,
            signature=response.signature,
        )
        result = gateway.verify(tampered)
        assert result.authenticated is False
        assert result.result == AuthResult.INVALID_NONCE


class TestExpiredChallenge:
    def test_expired_challenge_rejected(self):
        agent, gateway = make_registered_pair()
        challenge = gateway.issue_challenge("agent-001")
        response = agent.sign_challenge(challenge)

        # Manually expire the challenge
        state = gateway._pending_challenges[challenge.nonce]
        state.challenge = challenge.__class__(
            agent_id=challenge.agent_id,
            nonce=challenge.nonce,
            matrix=challenge.matrix,
            timestamp=0.0,  # epoch = definitely expired
        )
        result = gateway.verify(response)
        assert result.authenticated is False
        assert result.result == AuthResult.EXPIRED_CHALLENGE
