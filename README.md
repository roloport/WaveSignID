# WaveSignID

**Zero-trust digital biometrics for AI agents.**

Authenticate AI agents using optical cryptography — phase masks as identity, diffraction patterns as proof. Like a fingerprint, but for machines.

---

## What It Does

- **Identity** — Each agent gets a unique 2D phase mask (the "Digital Cornea") as its private key
- **Registration** — The agent publishes only its power spectrum — the phase is destroyed, making the key irreversible
- **Authentication** — Challenge-response: the agent proves it holds the correct phase mask without ever revealing it
- **Verification** — The gateway checks both a cryptographic signature and wave-optical correlation in a single pass

---

## How It Works

Two-layer architecture. Both layers must pass.

**Layer 1 — Wave Biometric** (what you are)

The agent's private key is a random 2D phase mask `M = e^{iφ}`. The public key is its power spectrum `|FFT(M)|²` — phase discarded. Recovering M from the power spectrum requires solving the **Phase Retrieval Problem** (NP-hard).

During authentication, the agent sends only `|FFT(C ⊙ M)|²` — magnitudes only, so M never leaks.

**Layer 2 — Cryptographic Signature** (what you know)

An Ed25519 signing key is derived from the phase mask via SHA3-256. The agent signs every response. The gateway verifies with the registered public key. Standard asymmetric crypto — the gateway holds no secrets.

### Protocol Flow

```
1. REGISTER   Agent → Gateway    {agent_id, power_spectrum, ed25519_verify_key}
2. CHALLENGE  Gateway → Agent    {nonce, random_matrix_C, timestamp}
3. RESPOND    Agent → Gateway    {|FFT(C ⊙ M)|², ed25519_signature, nonce}
4. VERIFY     Gateway checks     nonce fresh → signature valid → energy match → correlation pass
```

### Verification Pipeline

The gateway runs four sequential checks:

| Check | What It Does | Catches |
|-------|-------------|---------|
| Nonce freshness | Ensures challenge is single-use and not expired | Replay attacks |
| Ed25519 signature | Verifies cryptographic proof of identity | Forgery, tampering |
| Spectral energy | Parseval's theorem: total energy must match expected value | Random/fabricated proofs |
| Cross-correlation | Circular convolution of power spectra must correlate above threshold | Wrong phase mask |

---

## Security

| Attack | Defense |
|--------|---------|
| Recover private key from public key | Phase Retrieval is NP-hard — power spectrum destroys phase |
| Recover key from authentication traffic | Only magnitudes transmitted — phase is destroyed |
| Forge a wave proof without the key | Must solve constrained phase retrieval; statistical checks reject |
| Replay a previous response | Single-use nonce + 30-second expiry window |
| Forge the cryptographic signature | Ed25519 hardness (elliptic curve discrete log) |

---

## Quick Start

```bash
pip install -e .
```

```python
from waveid import WaveID_Agent, WaveID_Gateway

# Create identity
agent = WaveID_Agent("agent-001", mask_size=128)

# Register with gateway
gateway = WaveID_Gateway(mask_size=128)
gateway.register("agent-001", agent.get_public_key(), agent.get_verify_key())

# Authenticate
challenge = gateway.issue_challenge("agent-001")
response = agent.sign_challenge(challenge)
result = gateway.verify(response)

print(result.authenticated)       # True
print(result.correlation_score)   # ~0.82
```

---

## Demo

```bash
python demo.py
```

Runs three test cases and generates 3D correlation surface plots:

| Test | Scenario | Expected |
|------|----------|----------|
| 1 | Legitimate agent authenticates | Pass (correlation ~0.82) |
| 2 | Forged identity (wrong phase mask) | Rejected |
| 3 | Replay attack (reused response) | Rejected |

Output saved to `waveid_demo.png` — shows the sharp Dirac-like peak for valid auth vs. flat noise for forgery.

---

## Architecture

```
src/waveid/
├── crypto.py      # Wave primitives — phase mask, FFT, power spectrum, POC
├── identity.py    # WaveID_Agent — holds private mask, signs challenges
├── gateway.py     # WaveID_Gateway — registers agents, verifies responses
└── protocol.py    # Message types, nonce management, constants
tests/
├── test_crypto.py     # 22 tests — primitives, energy, correlation, POC
├── test_identity.py   # 7 tests — creation, signing, export/import
├── test_gateway.py    # 8 tests — registration, verification, replay, expiry
└── test_protocol.py   # 8 tests — full lifecycle, partial forgery, cleanup
demo.py                # Visual demo with 3D matplotlib plots
```

---

## Tests

```bash
pytest tests/ -v
```

45 tests covering:
- Correct authentication (true positive)
- Forgery detection (wrong phase mask)
- Replay prevention (reused nonce)
- Partial forgery (correct sig + wrong proof, and vice versa)
- Expired challenges, unknown agents, edge cases

---

## Dependencies

- `numpy` — matrix operations
- `scipy` — 2D FFT/IFFT
- `PyNaCl` — Ed25519 signatures (libsodium)
- `matplotlib` — visualization

---

## License

[Apache 2.0](./LICENSE)
