"""WaveID Agent — the entity that holds a private phase mask identity."""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from nacl.signing import SigningKey

from waveid.crypto import (
    compute_power_spectrum,
    compute_wave_proof,
    derive_signing_seed_from_angles,
    generate_phase_mask,
)
from waveid.protocol import (
    MASK_SIZE,
    Challenge,
    Response,
    build_signature_payload,
)


class WaveID_Agent:
    """Represents an AI agent's cryptographic identity.

    The agent holds a private 2D phase mask (the "Digital Cornea") which
    never leaves its enclave. The public key is the power spectrum of this
    mask — recovering the mask from the power spectrum requires solving
    the Phase Retrieval Problem (NP-hard).

    An Ed25519 signing key is deterministically derived from the phase mask
    so the mask serves as the single master secret.
    """

    def __init__(
        self,
        agent_id: str,
        mask_size: int = MASK_SIZE,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize agent identity by generating a fresh phase mask.

        Args:
            agent_id: Unique identifier for this agent.
            mask_size: Dimension N of the N×N phase mask.
            seed: Optional RNG seed (for testing reproducibility).
        """
        self._agent_id = agent_id
        self._mask_size = mask_size
        self._mask = generate_phase_mask(mask_size, seed=seed)
        # Store canonical phase angles to ensure deterministic key derivation
        # across export/import cycles (avoids float drift from np.angle roundtrips).
        self._phase_angles = np.angle(self._mask)

        # Derive Ed25519 keys from the canonical phase angles
        signing_seed = derive_signing_seed_from_angles(self._phase_angles)
        self._signing_key = SigningKey(signing_seed)
        self._verify_key = self._signing_key.verify_key

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def mask_size(self) -> int:
        return self._mask_size

    def get_public_key(self) -> np.ndarray:
        """Return the power spectrum |FFT(M)|² (safe to share publicly)."""
        return compute_power_spectrum(self._mask)

    def get_verify_key(self) -> bytes:
        """Return the Ed25519 public verification key as raw bytes."""
        return bytes(self._verify_key)

    def sign_challenge(self, challenge: Challenge) -> Response:
        """Produce a response to a gateway challenge.

        Computes:
        1. Wave proof: |FFT(C ⊙ M)|² (magnitude-only, does not leak M)
        2. Ed25519 signature over H(nonce ‖ C ‖ wave_proof)

        Args:
            challenge: The challenge issued by the gateway.

        Returns:
            A Response containing the wave proof and cryptographic signature.
        """
        wave_proof = compute_wave_proof(challenge.matrix, self._mask)

        payload = build_signature_payload(
            challenge.nonce, challenge.matrix, wave_proof
        )
        payload_hash = hashlib.sha3_256(payload).digest()
        signed = self._signing_key.sign(payload_hash)
        # signed.signature is the 64-byte detached signature
        signature = signed.signature

        return Response(
            agent_id=self._agent_id,
            nonce=challenge.nonce,
            wave_proof=wave_proof,
            signature=signature,
        )

    def export_identity(self) -> dict:
        """Serialize the agent's identity for storage.

        WARNING: The exported data contains the private phase mask.
        Encrypt at rest in production.
        """
        return {
            "agent_id": self._agent_id,
            "mask_size": self._mask_size,
            "phase_angles": self._phase_angles,
        }

    @classmethod
    def from_identity(cls, data: dict) -> WaveID_Agent:
        """Restore an agent from exported identity data."""
        agent = cls.__new__(cls)
        agent._agent_id = data["agent_id"]
        agent._mask_size = data["mask_size"]
        agent._phase_angles = data["phase_angles"]
        agent._mask = np.exp(1j * agent._phase_angles)

        signing_seed = derive_signing_seed_from_angles(agent._phase_angles)
        agent._signing_key = SigningKey(signing_seed)
        agent._verify_key = agent._signing_key.verify_key
        return agent
