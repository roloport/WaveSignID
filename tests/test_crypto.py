"""Unit tests for wave cryptography primitives."""

import numpy as np
import pytest

from waveid.crypto import (
    compute_power_spectrum,
    compute_wave_proof,
    derive_signing_seed,
    extract_phase_angles,
    generate_challenge_matrix,
    generate_phase_mask,
    phase_only_correlation,
    verify_cross_correlation,
    verify_spectral_energy,
)

N = 64  # Smaller mask for faster tests


class TestPhaseMask:
    def test_shape(self):
        mask = generate_phase_mask(N)
        assert mask.shape == (N, N)

    def test_unit_magnitude(self):
        mask = generate_phase_mask(N)
        magnitudes = np.abs(mask)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-12)

    def test_reproducible_with_seed(self):
        m1 = generate_phase_mask(N, seed=42)
        m2 = generate_phase_mask(N, seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_differ(self):
        m1 = generate_phase_mask(N, seed=1)
        m2 = generate_phase_mask(N, seed=2)
        assert not np.allclose(m1, m2)

    def test_extract_phase_roundtrip(self):
        mask = generate_phase_mask(N, seed=10)
        phi = extract_phase_angles(mask)
        reconstructed = np.exp(1j * phi)
        np.testing.assert_allclose(reconstructed, mask, atol=1e-12)


class TestPowerSpectrum:
    def test_shape(self):
        mask = generate_phase_mask(N, seed=1)
        pk = compute_power_spectrum(mask)
        assert pk.shape == (N, N)

    def test_non_negative(self):
        mask = generate_phase_mask(N, seed=1)
        pk = compute_power_spectrum(mask)
        assert np.all(pk >= 0)

    def test_different_masks_different_spectra(self):
        pk1 = compute_power_spectrum(generate_phase_mask(N, seed=1))
        pk2 = compute_power_spectrum(generate_phase_mask(N, seed=2))
        assert not np.allclose(pk1, pk2)

    def test_parseval_energy(self):
        """Parseval: sum(|FFT(M)|²) = N² * sum(|M|²) = N² * N² = N⁴."""
        mask = generate_phase_mask(N, seed=5)
        pk = compute_power_spectrum(mask)
        expected_energy = N ** 4  # |M|=1 so sum(|M|²) = N²
        np.testing.assert_allclose(np.sum(pk), expected_energy, rtol=1e-10)


class TestWaveProof:
    def test_shape(self):
        mask = generate_phase_mask(N, seed=1)
        C = generate_challenge_matrix(N)
        proof = compute_wave_proof(C, mask)
        assert proof.shape == (N, N)

    def test_non_negative(self):
        mask = generate_phase_mask(N, seed=1)
        C = generate_challenge_matrix(N)
        proof = compute_wave_proof(C, mask)
        assert np.all(proof >= 0)


class TestChallengeMatrix:
    def test_shape(self):
        C = generate_challenge_matrix(N)
        assert C.shape == (N, N)

    def test_value_range(self):
        C = generate_challenge_matrix(N)
        assert np.all(C >= 0.1)
        assert np.all(C <= 1.0)


class TestSpectralEnergyVerification:
    def test_valid_proof_passes(self):
        mask = generate_phase_mask(N, seed=1)
        pk = compute_power_spectrum(mask)
        C = generate_challenge_matrix(N)
        proof = compute_wave_proof(C, mask)
        assert verify_spectral_energy(proof, C, pk, tolerance=0.15)

    def test_random_proof_may_fail(self):
        """A random proof with wrong energy should fail."""
        mask = generate_phase_mask(N, seed=1)
        pk = compute_power_spectrum(mask)
        C = generate_challenge_matrix(N)
        fake_proof = np.random.uniform(0, 100, size=(N, N))
        # Very likely to have wrong energy
        result = verify_spectral_energy(fake_proof, C, pk, tolerance=0.01)
        assert not result


class TestCrossCorrelation:
    def test_valid_proof_has_high_correlation(self):
        mask = generate_phase_mask(N, seed=1)
        pk = compute_power_spectrum(mask)
        C = generate_challenge_matrix(N)
        proof = compute_wave_proof(C, mask)
        passed, score = verify_cross_correlation(proof, C, pk, threshold=0.3)
        assert passed
        assert score > 0.3

    def test_wrong_mask_has_low_correlation(self):
        real_mask = generate_phase_mask(N, seed=1)
        fake_mask = generate_phase_mask(N, seed=999)
        pk = compute_power_spectrum(real_mask)
        C = generate_challenge_matrix(N)
        fake_proof = compute_wave_proof(C, fake_mask)
        passed, score = verify_cross_correlation(fake_proof, C, pk, threshold=0.5)
        assert not passed


class TestDeriveSigningSeed:
    def test_deterministic(self):
        mask = generate_phase_mask(N, seed=42)
        s1 = derive_signing_seed(mask)
        s2 = derive_signing_seed(mask)
        assert s1 == s2

    def test_length(self):
        mask = generate_phase_mask(N, seed=42)
        seed = derive_signing_seed(mask)
        assert len(seed) == 32

    def test_different_masks_different_seeds(self):
        m1 = generate_phase_mask(N, seed=1)
        m2 = generate_phase_mask(N, seed=2)
        assert derive_signing_seed(m1) != derive_signing_seed(m2)


class TestPhaseOnlyCorrelation:
    def test_identical_signals_sharp_peak(self):
        mask = generate_phase_mask(N, seed=1)
        poc = phase_only_correlation(mask, mask)
        peak = np.max(poc)
        mean = np.mean(poc)
        # Peak should be much higher than mean for identical signals
        assert peak > 5 * mean

    def test_different_signals_no_peak(self):
        m1 = generate_phase_mask(N, seed=1)
        m2 = generate_phase_mask(N, seed=2)
        poc = phase_only_correlation(m1, m2)
        peak = np.max(poc)
        mean = np.mean(poc)
        # No dominant peak for different signals
        assert peak < 5 * mean
