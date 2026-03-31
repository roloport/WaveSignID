"""Low-level wave cryptography primitives.

Implements phase mask generation, 2D FFT operations, power spectrum
computation, and Phase-Only Correlation (POC) for the WaveID protocol.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from scipy.fft import fft2, ifft2

# Small constant to prevent division by zero in spectral operations.
_EPSILON = 1e-10


def generate_phase_mask(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random 2D phase mask M = e^{iφ}.

    Args:
        n: Size of the N×N phase mask.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Complex N×N array where each element has unit magnitude and
        uniformly random phase in [0, 2π).
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, size=(n, n))
    return np.exp(1j * phi)


def extract_phase_angles(mask: np.ndarray) -> np.ndarray:
    """Extract the phase angles φ from a phase mask M = e^{iφ}.

    Args:
        mask: Complex N×N phase mask.

    Returns:
        Real N×N array of phase angles in (-π, π].
    """
    return np.angle(mask)


def compute_power_spectrum(mask: np.ndarray) -> np.ndarray:
    """Compute the power spectrum |FFT(M)|² (the public key).

    The phase information is destroyed — only magnitudes squared remain.
    Recovering the original mask from the power spectrum requires solving
    the Phase Retrieval Problem, which is NP-hard in the general case.

    Args:
        mask: Complex N×N phase mask (private key).

    Returns:
        Real N×N power spectrum array (public key).
    """
    spectrum = fft2(mask)
    return np.abs(spectrum) ** 2


def compute_wave_proof(
    challenge: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Compute the wave proof |FFT(C ⊙ M)|² for a challenge.

    The element-wise product modulates the challenge by the agent's
    private phase mask, then the magnitude-squared of the FFT is taken
    so the phase (which encodes M) is destroyed.

    Args:
        challenge: Real N×N challenge matrix from the gateway.
        mask: Complex N×N phase mask (private key).

    Returns:
        Real N×N wave proof array (magnitude-only, safe to transmit).
    """
    modulated = challenge * mask
    spectrum = fft2(modulated)
    return np.abs(spectrum) ** 2


def generate_challenge_matrix(n: int) -> np.ndarray:
    """Generate a random real-valued challenge matrix.

    Values are in [0.1, 1.0] to ensure non-zero entries and prevent
    degenerate modulation.

    Args:
        n: Size of the N×N matrix.

    Returns:
        Real N×N challenge matrix.
    """
    return np.random.uniform(0.1, 1.0, size=(n, n))


def verify_spectral_energy(
    wave_proof: np.ndarray,
    challenge: np.ndarray,
    public_key: np.ndarray,
    tolerance: float = 0.15,
) -> bool:
    """Check that the total spectral energy of the proof is consistent.

    By Parseval's theorem, the total energy of |FFT(C ⊙ M)|² should be
    proportional to the energy of the input signal C ⊙ M. For a valid
    proof, the ratio of observed to expected energy should be close to 1.

    Specifically:
        sum(|FFT(C ⊙ M)|²) = N² · sum(|C ⊙ M|²) = N² · sum(C² · |M|²)
    Since |M|² = 1 for a phase mask:
        expected_energy = N² · sum(C²)

    Args:
        wave_proof: The |FFT(C ⊙ M)|² submitted by the agent.
        challenge: The challenge matrix C issued by the gateway.
        public_key: The registered power spectrum (not used in energy
            check but included for API consistency).
        tolerance: Acceptable relative deviation from expected energy.

    Returns:
        True if the energy ratio is within [1-tolerance, 1+tolerance].
    """
    n = challenge.shape[0]
    observed_energy = float(np.sum(wave_proof))
    # Parseval: sum(|FFT(x)|^2) = N^2 * sum(|x|^2)
    # For x = C * M where |M|=1: sum(|x|^2) = sum(C^2)
    expected_energy = float((n ** 2) * np.sum(challenge ** 2))
    if expected_energy < _EPSILON:
        return False
    ratio = observed_energy / expected_energy
    return bool(abs(ratio - 1.0) <= tolerance)


def verify_cross_correlation(
    wave_proof: np.ndarray,
    challenge: np.ndarray,
    public_key: np.ndarray,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Verify the wave proof via cross-correlation with expected pattern.

    Element-wise product in spatial domain (C ⊙ M) corresponds to circular
    convolution in frequency domain: FFT(C⊙M) = FFT(C) ⊛ FFT(M) / N².

    The expected value of |FFT(C⊙M)|² at frequency k is:
        E[|FFT(C⊙M)|²_k] = (1/N⁴) Σ_j |FFT(C)_{k-j}|² · |FFT(M)_j|²

    This is the circular convolution of the two power spectra, normalized
    by N⁴. We compute this as the reference pattern and measure normalized
    correlation against the observed wave proof.

    Args:
        wave_proof: The |FFT(C ⊙ M)|² from the agent.
        challenge: The challenge matrix C.
        public_key: The registered |FFT(M)|² (power spectrum).
        threshold: Minimum correlation score to accept.

    Returns:
        Tuple of (passed, correlation_score).
    """
    n = challenge.shape[0]

    # Compute power spectrum of the challenge
    challenge_ps = np.abs(fft2(challenge)) ** 2

    # Reference = circular convolution of power spectra / N⁴
    # By convolution theorem: conv(A, B) = IFFT(FFT(A) · FFT(B))
    reference = np.real(ifft2(fft2(challenge_ps) * fft2(public_key))) / (n ** 4)

    # Normalize both to zero-mean, unit-variance for Pearson correlation
    proof_flat = wave_proof.ravel().astype(np.float64)
    ref_flat = reference.ravel().astype(np.float64)

    proof_centered = proof_flat - np.mean(proof_flat)
    ref_centered = ref_flat - np.mean(ref_flat)

    proof_norm = np.linalg.norm(proof_centered)
    ref_norm = np.linalg.norm(ref_centered)

    if proof_norm < _EPSILON or ref_norm < _EPSILON:
        return False, 0.0

    correlation = float(np.dot(proof_centered, ref_centered) / (proof_norm * ref_norm))

    return bool(correlation >= threshold), correlation


def derive_signing_seed(mask: np.ndarray) -> bytes:
    """Derive a 32-byte Ed25519 signing seed from a phase mask.

    Uses SHA3-256 of the serialized phase angles. Prefer
    derive_signing_seed_from_angles() when you already have the
    canonical angle array, to avoid float drift from np.angle().

    Args:
        mask: Complex N×N phase mask.

    Returns:
        32-byte seed suitable for Ed25519 key generation.
    """
    return derive_signing_seed_from_angles(np.angle(mask))


def derive_signing_seed_from_angles(phase_angles: np.ndarray) -> bytes:
    """Derive a 32-byte Ed25519 signing seed from phase angles.

    Args:
        phase_angles: Real N×N array of phase angles.

    Returns:
        32-byte seed suitable for Ed25519 key generation.
    """
    return hashlib.sha3_256(phase_angles.tobytes()).digest()


def phase_only_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Phase-Only Correlation (POC) between two 2D signals.

    POC normalizes the cross-power spectrum by its magnitude before
    applying IFFT, producing a sharp Dirac-like peak when the inputs
    match and flat noise otherwise.

    Args:
        a: First 2D complex or real array.
        b: Second 2D complex or real array.

    Returns:
        Real 2D correlation surface. A sharp peak indicates match.
    """
    A = fft2(a)
    B = fft2(b)
    cross_power = A * np.conj(B)
    magnitude = np.abs(cross_power) + _EPSILON
    normalized = cross_power / magnitude
    return np.abs(ifft2(normalized))
