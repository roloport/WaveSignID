"""Protocol message types, constants, and challenge state management."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# --- Protocol Constants ---

MASK_SIZE: int = 128
NONCE_BYTES: int = 32
NONCE_EXPIRY_SECONDS: float = 30.0
ENERGY_TOLERANCE: float = 0.15
CORRELATION_THRESHOLD: float = 0.5


class AuthResult(Enum):
    """Outcome of an authentication attempt."""
    SUCCESS = "success"
    INVALID_NONCE = "invalid_nonce"
    EXPIRED_CHALLENGE = "expired_challenge"
    INVALID_SIGNATURE = "invalid_signature"
    ENERGY_MISMATCH = "energy_mismatch"
    CORRELATION_FAILURE = "correlation_failure"
    UNKNOWN_AGENT = "unknown_agent"


@dataclass(frozen=True)
class RegistrationRequest:
    """Data sent by an agent during registration."""
    agent_id: str
    public_key: np.ndarray      # |FFT(M)|² power spectrum
    verify_key: bytes            # Ed25519 public verify key


@dataclass(frozen=True)
class Challenge:
    """Challenge issued by the gateway to an agent."""
    agent_id: str
    nonce: bytes                 # Random nonce for replay prevention
    matrix: np.ndarray           # N×N challenge matrix C
    timestamp: float             # Unix timestamp of issuance


@dataclass(frozen=True)
class Response:
    """Response from an agent to a challenge."""
    agent_id: str
    nonce: bytes                 # Echo of the challenge nonce
    wave_proof: np.ndarray       # |FFT(C ⊙ M)|²
    signature: bytes             # Ed25519 signature over H(nonce ‖ C ‖ wave_proof)


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying an agent's response."""
    authenticated: bool
    result: AuthResult
    correlation_score: Optional[float] = None
    details: str = ""


@dataclass
class ChallengeState:
    """Tracks an outstanding challenge for replay prevention."""
    challenge: Challenge
    used: bool = False

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this challenge has exceeded the expiry window."""
        if now is None:
            now = time.time()
        return (now - self.challenge.timestamp) > NONCE_EXPIRY_SECONDS

    def mark_used(self) -> None:
        """Mark this challenge as consumed (single-use)."""
        self.used = True


def generate_nonce() -> bytes:
    """Generate a cryptographically random nonce."""
    return os.urandom(NONCE_BYTES)


def build_signature_payload(
    nonce: bytes,
    challenge_matrix: np.ndarray,
    wave_proof: np.ndarray,
) -> bytes:
    """Build the canonical byte payload for signing/verification.

    Args:
        nonce: The challenge nonce.
        challenge_matrix: The N×N challenge matrix.
        wave_proof: The |FFT(C ⊙ M)|² proof array.

    Returns:
        Concatenated bytes: nonce ‖ challenge_bytes ‖ proof_bytes.
    """
    return nonce + challenge_matrix.tobytes() + wave_proof.tobytes()
