#!/usr/bin/env python3
"""WaveID Demo — Visual demonstration of the authentication lifecycle.

Runs three test cases:
1. True Positive:  Legitimate agent authenticates successfully.
2. True Negative:  Forged identity (wrong phase mask) is rejected.
3. True Negative:  Replay attack (reused response) is rejected.

Generates 3D surface plots showing the correlation peak (or lack thereof)
for each case.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from waveid.crypto import compute_wave_proof, generate_phase_mask, phase_only_correlation
from waveid.gateway import WaveID_Gateway
from waveid.identity import WaveID_Agent
from waveid.protocol import AuthResult

MASK_SIZE = 128


def plot_correlation_surface(
    ax: Axes3D,
    data: np.ndarray,
    title: str,
    color: str = "viridis",
) -> None:
    """Plot a 3D surface of a 2D correlation map (center region)."""
    # Show center 32x32 region for clarity
    n = data.shape[0]
    center = n // 2
    half = 16
    region = np.fft.fftshift(data)[center - half:center + half, center - half:center + half]

    x = np.arange(region.shape[0])
    y = np.arange(region.shape[1])
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, region.T, cmap=color, edgecolor="none", alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_zlabel("Intensity")


def run_demo() -> None:
    print("=" * 60)
    print("  WaveID — Zero-Trust Digital Biometrics for AI Agents")
    print("=" * 60)
    print()

    # --- Setup ---
    print("[1/4] Creating agent identity (128x128 phase mask)...")
    agent = WaveID_Agent("alice-prime", mask_size=MASK_SIZE, seed=42)
    gateway = WaveID_Gateway(mask_size=MASK_SIZE)

    print("[2/4] Registering agent with gateway...")
    gateway.register(
        agent.agent_id,
        agent.get_public_key(),
        agent.get_verify_key(),
    )
    print(f"       Registered: {agent.agent_id}")
    print(f"       Public key shape: {agent.get_public_key().shape}")
    print(f"       Verify key: {agent.get_verify_key().hex()[:32]}...")
    print()

    # --- Test Case 1: True Positive ---
    print("-" * 60)
    print("TEST 1: Legitimate Authentication (True Positive)")
    print("-" * 60)
    challenge = gateway.issue_challenge(agent.agent_id)
    response = agent.sign_challenge(challenge)
    result = gateway.verify(response)

    print(f"  Result:      {result.result.value}")
    print(f"  Correlation: {result.correlation_score:.4f}")
    print(f"  Auth:        {'PASS' if result.authenticated else 'FAIL'}")
    assert result.authenticated, "Test 1 failed!"
    print()

    # Compute POC for visualization (agent's mask vs itself)
    poc_valid = phase_only_correlation(agent._mask, agent._mask)

    # --- Test Case 2: True Negative (Forgery) ---
    print("-" * 60)
    print("TEST 2: Forged Identity — Wrong Phase Mask (True Negative)")
    print("-" * 60)
    fake_agent = WaveID_Agent("alice-prime", mask_size=MASK_SIZE, seed=999)

    challenge2 = gateway.issue_challenge(agent.agent_id)
    fake_response = fake_agent.sign_challenge(challenge2)
    result2 = gateway.verify(fake_response)

    print(f"  Result:      {result2.result.value}")
    if result2.correlation_score is not None:
        print(f"  Correlation: {result2.correlation_score:.4f}")
    print(f"  Auth:        {'PASS' if result2.authenticated else 'FAIL'}")
    assert not result2.authenticated, "Test 2 failed — forgery should be rejected!"
    print()

    # POC for visualization (real mask vs fake mask)
    poc_forged = phase_only_correlation(agent._mask, fake_agent._mask)

    # --- Test Case 3: True Negative (Replay Attack) ---
    print("-" * 60)
    print("TEST 3: Replay Attack — Reused Response (True Negative)")
    print("-" * 60)
    challenge3 = gateway.issue_challenge(agent.agent_id)
    response3 = agent.sign_challenge(challenge3)

    # First use
    result3a = gateway.verify(response3)
    print(f"  First use:   {result3a.result.value} ({'PASS' if result3a.authenticated else 'FAIL'})")
    assert result3a.authenticated, "First use should pass!"

    # Replay
    result3b = gateway.verify(response3)
    print(f"  Replay:      {result3b.result.value} ({'PASS' if result3b.authenticated else 'FAIL'})")
    assert not result3b.authenticated, "Replay should be rejected!"
    print()

    # --- Visualization ---
    print("-" * 60)
    print("Generating 3D correlation surface plots...")
    print("-" * 60)

    fig = plt.figure(figsize=(16, 5))

    # Plot 1: Public key power spectrum
    ax1 = fig.add_subplot(131)
    pk = agent.get_public_key()
    pk_shifted = np.fft.fftshift(np.log1p(pk))
    ax1.imshow(pk_shifted, cmap="inferno")
    ax1.set_title("Public Key\n(Log Power Spectrum)", fontweight="bold")
    ax1.set_xlabel("kx")
    ax1.set_ylabel("ky")

    # Plot 2: POC — Valid (sharp peak)
    ax2 = fig.add_subplot(132, projection="3d")
    plot_correlation_surface(ax2, poc_valid, "POC: Valid Identity\n(Sharp Dirac Peak)", "plasma")

    # Plot 3: POC — Forged (flat noise)
    ax3 = fig.add_subplot(133, projection="3d")
    plot_correlation_surface(ax3, poc_forged, "POC: Forged Identity\n(Flat White Noise)", "coolwarm")

    plt.tight_layout()
    plt.savefig("waveid_demo.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: waveid_demo.png")
    print()

    # --- Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  Test 1 (Legitimate):  PASS")
    print("  Test 2 (Forgery):     REJECTED (as expected)")
    print("  Test 3 (Replay):      REJECTED (as expected)")
    print()
    print("  All tests passed. WaveID protocol is functioning correctly.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
