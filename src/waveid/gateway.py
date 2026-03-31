"""WaveID Gateway — stateless API middleware that verifies agent identities."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

from waveid.crypto import (
    generate_challenge_matrix,
    verify_cross_correlation,
    verify_spectral_energy,
)
from waveid.protocol import (
    CORRELATION_THRESHOLD,
    ENERGY_TOLERANCE,
    MASK_SIZE,
    AuthResult,
    Challenge,
    ChallengeState,
    Response,
    VerificationResult,
    build_signature_payload,
    generate_nonce,
)


@dataclass
class _AgentRecord:
    """Internal registry entry for a registered agent."""
    agent_id: str
    public_key: np.ndarray   # |FFT(M)|² power spectrum
    verify_key: VerifyKey     # Ed25519 verify key
    mask_size: int


class WaveID_Gateway:
    """Gateway that authenticates agents via the WaveID protocol.

    Maintains a registry of agents (public keys + verify keys) and
    manages challenge-response authentication. Each challenge is
    single-use and time-bounded to prevent replay attacks.
    """

    def __init__(
        self,
        mask_size: int = MASK_SIZE,
        energy_tolerance: float = ENERGY_TOLERANCE,
        correlation_threshold: float = CORRELATION_THRESHOLD,
    ) -> None:
        self._mask_size = mask_size
        self._energy_tolerance = energy_tolerance
        self._correlation_threshold = correlation_threshold
        self._registry: dict[str, _AgentRecord] = {}
        self._pending_challenges: dict[bytes, ChallengeState] = {}

    @property
    def registered_agents(self) -> list[str]:
        """List of registered agent IDs."""
        return list(self._registry.keys())

    def register(
        self,
        agent_id: str,
        public_key: np.ndarray,
        verify_key: bytes,
    ) -> None:
        """Register an agent's public identity.

        Args:
            agent_id: Unique identifier for the agent.
            public_key: The agent's power spectrum |FFT(M)|².
            verify_key: Raw bytes of the agent's Ed25519 public key.
        """
        self._registry[agent_id] = _AgentRecord(
            agent_id=agent_id,
            public_key=public_key,
            verify_key=VerifyKey(verify_key),
            mask_size=self._mask_size,
        )

    def issue_challenge(self, agent_id: str) -> Challenge:
        """Generate a fresh challenge for an agent.

        Args:
            agent_id: The agent to challenge.

        Returns:
            A Challenge containing a random nonce and matrix.

        Raises:
            KeyError: If the agent_id is not registered.
        """
        if agent_id not in self._registry:
            raise KeyError(f"Unknown agent: {agent_id}")

        nonce = generate_nonce()
        matrix = generate_challenge_matrix(self._mask_size)
        challenge = Challenge(
            agent_id=agent_id,
            nonce=nonce,
            matrix=matrix,
            timestamp=time.time(),
        )
        self._pending_challenges[nonce] = ChallengeState(challenge=challenge)
        return challenge

    def verify(self, response: Response) -> VerificationResult:
        """Verify an agent's response to a challenge.

        Runs four checks in sequence:
        1. Nonce freshness (replay prevention)
        2. Ed25519 signature validity
        3. Spectral energy consistency
        4. Cross-correlation with expected pattern

        Args:
            response: The agent's response.

        Returns:
            VerificationResult indicating success or specific failure.
        """
        # --- Check 0: Agent exists ---
        record = self._registry.get(response.agent_id)
        if record is None:
            return VerificationResult(
                authenticated=False,
                result=AuthResult.UNKNOWN_AGENT,
                details=f"Agent '{response.agent_id}' is not registered.",
            )

        # --- Check 1: Nonce freshness ---
        state = self._pending_challenges.get(response.nonce)
        if state is None or state.used:
            return VerificationResult(
                authenticated=False,
                result=AuthResult.INVALID_NONCE,
                details="Nonce not found or already used.",
            )
        if state.is_expired():
            state.mark_used()
            return VerificationResult(
                authenticated=False,
                result=AuthResult.EXPIRED_CHALLENGE,
                details="Challenge has expired.",
            )
        # Consume the nonce (single-use)
        challenge = state.challenge
        state.mark_used()

        # --- Check 2: Ed25519 signature ---
        payload = build_signature_payload(
            response.nonce, challenge.matrix, response.wave_proof
        )
        payload_hash = hashlib.sha3_256(payload).digest()
        try:
            record.verify_key.verify(payload_hash, response.signature)
        except BadSignatureError:
            return VerificationResult(
                authenticated=False,
                result=AuthResult.INVALID_SIGNATURE,
                details="Ed25519 signature verification failed.",
            )

        # --- Check 3: Spectral energy ---
        energy_ok = verify_spectral_energy(
            response.wave_proof,
            challenge.matrix,
            record.public_key,
            tolerance=self._energy_tolerance,
        )
        if not energy_ok:
            return VerificationResult(
                authenticated=False,
                result=AuthResult.ENERGY_MISMATCH,
                details="Wave proof spectral energy out of expected range.",
            )

        # --- Check 4: Cross-correlation ---
        corr_ok, corr_score = verify_cross_correlation(
            response.wave_proof,
            challenge.matrix,
            record.public_key,
            threshold=self._correlation_threshold,
        )
        if not corr_ok:
            return VerificationResult(
                authenticated=False,
                result=AuthResult.CORRELATION_FAILURE,
                correlation_score=corr_score,
                details=f"Cross-correlation {corr_score:.4f} below threshold.",
            )

        return VerificationResult(
            authenticated=True,
            result=AuthResult.SUCCESS,
            correlation_score=corr_score,
            details="All checks passed.",
        )

    def cleanup_expired(self) -> int:
        """Remove expired challenge states. Returns count removed."""
        now = time.time()
        expired = [
            nonce for nonce, state in self._pending_challenges.items()
            if state.is_expired(now) or state.used
        ]
        for nonce in expired:
            del self._pending_challenges[nonce]
        return len(expired)
