"""Integration tests for the full WaveID protocol flow."""

import hashlib

import numpy as np
import pytest
from nacl.signing import SigningKey

from waveid.crypto import (
    compute_wave_proof,
    generate_challenge_matrix,
    generate_phase_mask,
)
from waveid.gateway import WaveID_Gateway
from waveid.identity import WaveID_Agent
from waveid.protocol import (
    AuthResult,
    Response,
    build_signature_payload,
    generate_nonce,
)

N = 64


class TestFullProtocolLifecycle:
    """End-to-end protocol tests."""

    def test_register_challenge_respond_verify(self):
        """Happy path: full lifecycle succeeds."""
        agent = WaveID_Agent("alice", mask_size=N, seed=42)
        gateway = WaveID_Gateway(mask_size=N)

        # Register
        gateway.register("alice", agent.get_public_key(), agent.get_verify_key())

        # Challenge → Respond → Verify
        challenge = gateway.issue_challenge("alice")
        response = agent.sign_challenge(challenge)
        result = gateway.verify(response)

        assert result.authenticated is True
        assert result.result == AuthResult.SUCCESS

    def test_multiple_agents(self):
        """Multiple agents can register and authenticate independently."""
        gateway = WaveID_Gateway(mask_size=N)
        agents = [WaveID_Agent(f"agent-{i}", mask_size=N, seed=i) for i in range(5)]

        for agent in agents:
            gateway.register(agent.agent_id, agent.get_public_key(), agent.get_verify_key())

        for agent in agents:
            challenge = gateway.issue_challenge(agent.agent_id)
            response = agent.sign_challenge(challenge)
            result = gateway.verify(response)
            assert result.authenticated is True, f"{agent.agent_id} failed auth"

    def test_sequential_challenges_same_agent(self):
        """Same agent can authenticate multiple times with fresh challenges."""
        agent = WaveID_Agent("alice", mask_size=N, seed=42)
        gateway = WaveID_Gateway(mask_size=N)
        gateway.register("alice", agent.get_public_key(), agent.get_verify_key())

        for _ in range(5):
            challenge = gateway.issue_challenge("alice")
            response = agent.sign_challenge(challenge)
            result = gateway.verify(response)
            assert result.authenticated is True


class TestPartialForgery:
    """Tests where the attacker has some but not all correct components."""

    def test_correct_sig_wrong_wave_proof(self):
        """Attacker has the right Ed25519 key but wrong phase mask.

        This scenario: attacker stole the Ed25519 private key but not the
        phase mask. They can produce a valid signature but the wave proof
        will fail statistical checks.
        """
        agent = WaveID_Agent("alice", mask_size=N, seed=42)
        gateway = WaveID_Gateway(mask_size=N)
        gateway.register("alice", agent.get_public_key(), agent.get_verify_key())

        challenge = gateway.issue_challenge("alice")

        # Attacker generates a fake wave proof with a wrong mask
        fake_mask = generate_phase_mask(N, seed=999)
        fake_wave_proof = compute_wave_proof(challenge.matrix, fake_mask)

        # But signs it with the real key (simulating stolen Ed25519 key)
        # We need to access the agent's signing key for this test
        from waveid.crypto import derive_signing_seed_from_angles
        real_seed = derive_signing_seed_from_angles(agent._phase_angles)
        real_signing_key = SigningKey(real_seed)

        payload = build_signature_payload(
            challenge.nonce, challenge.matrix, fake_wave_proof
        )
        payload_hash = hashlib.sha3_256(payload).digest()
        signature = real_signing_key.sign(payload_hash).signature

        forged_response = Response(
            agent_id="alice",
            nonce=challenge.nonce,
            wave_proof=fake_wave_proof,
            signature=signature,
        )
        result = gateway.verify(forged_response)
        assert result.authenticated is False
        # Should fail at correlation check (sig is valid but proof doesn't match)
        assert result.result in (
            AuthResult.ENERGY_MISMATCH,
            AuthResult.CORRELATION_FAILURE,
        )

    def test_correct_wave_proof_wrong_sig(self):
        """Attacker has the right phase mask but wrong Ed25519 key.

        This scenario: attacker cloned the phase mask but not the signing
        key. Wave proof would be correct but signature will fail.
        """
        agent = WaveID_Agent("alice", mask_size=N, seed=42)
        gateway = WaveID_Gateway(mask_size=N)
        gateway.register("alice", agent.get_public_key(), agent.get_verify_key())

        challenge = gateway.issue_challenge("alice")

        # Compute a valid wave proof with the real mask
        real_wave_proof = compute_wave_proof(challenge.matrix, agent._mask)

        # Sign with a different key
        wrong_key = SigningKey.generate()
        payload = build_signature_payload(
            challenge.nonce, challenge.matrix, real_wave_proof
        )
        payload_hash = hashlib.sha3_256(payload).digest()
        wrong_sig = wrong_key.sign(payload_hash).signature

        forged_response = Response(
            agent_id="alice",
            nonce=challenge.nonce,
            wave_proof=real_wave_proof,
            signature=wrong_sig,
        )
        result = gateway.verify(forged_response)
        assert result.authenticated is False
        assert result.result == AuthResult.INVALID_SIGNATURE


class TestCleanup:
    def test_cleanup_expired_challenges(self):
        agent = WaveID_Agent("alice", mask_size=N, seed=42)
        gateway = WaveID_Gateway(mask_size=N)
        gateway.register("alice", agent.get_public_key(), agent.get_verify_key())

        # Issue several challenges
        for _ in range(10):
            gateway.issue_challenge("alice")

        assert len(gateway._pending_challenges) == 10

        # Manually expire them all
        for state in gateway._pending_challenges.values():
            state.challenge = state.challenge.__class__(
                agent_id=state.challenge.agent_id,
                nonce=state.challenge.nonce,
                matrix=state.challenge.matrix,
                timestamp=0.0,
            )

        removed = gateway.cleanup_expired()
        assert removed == 10
        assert len(gateway._pending_challenges) == 0
