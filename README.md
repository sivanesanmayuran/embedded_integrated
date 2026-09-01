# embedded_integrated

An integrated embedded library and demo firmware that brings together three
domains commonly required in modern edge / signal-processing devices:

- **DSP** — lightweight, allocation-free digital filters (FIR, IIR biquad,
  moving averages) plus a polyphase-filter-bank (PFB) channelizer.
- **AI / ML** — on-device inference using **TensorFlow Lite for
  Microcontrollers (TFLM)**.
- **Crypto** — **post-quantum** key encapsulation using **PQClean's
  ML-KEM-768** (the NIST-standardized Kyber KEM).

Everything is wired together through a single, modular `Makefile` that can
build a native **host** binary or **cross-compile** for ARM Cortex-M targets.
A companion Python model (`app/channelizer.py`) reproduces and visualizes the
DSP channelizer for verification.

---

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Components](#components)
  - [DSP Library](#1-dsp-library-libdsp)
  - [ML / TFLM](#2-ml--tflm-libmltflm)
  - [Post-Quantum Crypto](#3-post-quantum-crypto-libcryptopqclean)
  - [Channelizer Simulation](#4-channelizer-simulation)
- [Building](#building)
- [Running](#running)
- [Cross-Compiling for ARM](#cross-compiling-for-arm)
- [Regenerating the ML Model](#regenerating-the-ml-model)
- [Requirements](#requirements)
- [License](#license)

---

## Features

| Domain | What it does |
| ------ | ------------ |
| DSP    | FIR filter, IIR biquad (Direct Form II Transposed), simple/exponential moving averages, 16-channel polyphase channelizer with a DVB-S2 16-APSK modem demo |
| AI/ML  | Runs a TFLite model on-device via TFLM (`y = 2x + 1` demo model) |
| Crypto | ML-KEM-768 keypair / encapsulate / decapsulate round-trip (post-quantum) |
| Build  | One `Makefile`, host or ARM cross-build, dead-code stripping, size report |
| Sim    | NumPy/Matplotlib reference of the channelizer with constellation plots |

---

## Repository Layout

```
embedded_integrated/
├── Makefile                 # Consolidated, modular build system
├── app/                     # Application / demo entry point
│   ├── main.c               # Runs the PQC, DSP, channelizer and TFLM demos
│   ├── channelizer.py       # Python reference + plots for the PFB channelizer
│   ├── gen_model_data.py    # Trains/exports the TFLite model to C arrays
│   ├── model_data.{h,cc}    # Generated TFLite model as a byte array
│   └── tflm_example.{h,cc}  # TFLM glue: TFLM_Init() / TFLM_Run()
├── include/                 # Shared / public headers
├── lib/
│   ├── crypto/pqclean/       # PQClean ML-KEM-768 (clean C reference)
│   ├── dsp/                  # FIR / IIR / moving-average filters
│   │   ├── fir_filter.{c,h}
│   │   ├── iir_filter.{c,h}
│   │   └── moving_avg.{c,h}
│   ├── ml/tflm/              # TensorFlow Lite for Microcontrollers tree
│   └── utils/
└── build/                   # Build artifacts (bin/, obj/) — generated
```

---

## Components

### 1. DSP Library (`lib/dsp`)

Small, dependency-free C99 filters designed for embedded use — **no dynamic
allocation, no stdlib beyond math**. Each has a compact struct plus
`init` / `reset` / `update` / `update_block` functions.

- **FIR filter** (`fir_filter.h`) — up to `FIR_MAX_TAPS` (128) taps, ideal for
  linear-phase low/high/band-pass filtering.
- **IIR biquad** (`iir_filter.h`) — 2nd-order Direct Form II Transposed section
  (coefficients `b0,b1,b2,a1,a2` already normalized by `a0`).
- **Moving averages** (`moving_avg.h`) — Simple Moving Average (windowed) and
  Exponential Moving Average.

The `dsp_tests()` routine in `main.c` demonstrates a 16-tap Hamming-windowed
low-pass FIR and a 2nd-order low-pass IIR biquad against a unit-step input.

### 2. ML / TFLM (`lib/ml/tflm`)

Runs on-device inference with **TensorFlow Lite for Microcontrollers**. The
application exposes two C entry points:

- `TFLM_Init()` — sets up the interpreter, ops resolver and tensor arena.
- `TFLM_Run(float input, float *output)` — runs a single inference.

The bundled model (`model_data.cc`) is a trivial fully-connected network that
computes `y = 2*x + 1`, kept minimal so the pipeline can be verified end to
end. The `Makefile` compiles the required TFLM subtrees while filtering out
tests, benchmarks and example `main()`s.

### 3. Post-Quantum Crypto (`lib/crypto/pqclean`)

Uses the **clean, portable C** reference implementation of **ML-KEM-768**
(Kyber) from PQClean. `pqclean_test()` in `main.c` performs a full KEM
round-trip:

1. `crypto_kem_keypair()` — generate public/secret keys.
2. `crypto_kem_enc()` — encapsulate: produce a ciphertext and a shared secret.
3. `crypto_kem_dec()` — decapsulate: recover the shared secret from ciphertext.
4. Verify both sides derived the identical 32-byte shared secret.

### 4. Channelizer Simulation

A **16-channel polyphase filter bank (PFB) channelizer** is implemented in
both C (`main.c`) and Python (`app/channelizer.py`):

- Builds a DVB-S2-style **16-APSK** constellation (4 inner + 12 outer points).
- Synthesizes a multi-carrier wideband signal (IDFT) from per-channel symbols.
- Recovers individual channels via a polyphase prototype low-pass filter plus a
  16-point DFT demodulation backend.

The Python version (`channelizer.py`) streams 100 symbols and plots the
wideband time-domain signal, its spectrum, and the recovered 16-APSK
constellations for channels 2 and 5 — a convenient golden reference for the C
implementation.

---

## Building

The default target is a **native host** build:

```bash
make                # build -> build/bin/app
make run            # build then run the host binary
make clean          # remove the build/ directory
make print-vars     # debug: dump computed source lists and flags
```

Common overrides:

```bash
make OPT=-O2        # change optimization level (default -Os)
```

---

## Running

```bash
make run
```

The demo runs each subsystem in sequence and prints:

- The ML-KEM-768 shared-secret match result.
- The recovered channelizer baseband channels.
- FIR and IIR filter step responses.
- The TFLM inference output.

---

## Cross-Compiling for ARM

The build system supports ARM Cortex-M cross-compilation with a configurable
toolchain and CPU/FPU:

```bash
# Default: Cortex-M4F, hard-float, arm-none-eabi- toolchain
make PLATFORM=arm

# Override core / FPU / float ABI
make PLATFORM=arm ARM_CPU=cortex-m7 ARM_FPU=fpv5-d16 ARM_FLOAT_ABI=hard

# Override the toolchain prefix
make PLATFORM=arm CROSS_COMPILE=arm-none-eabi-
```

The ARM build links against newlib nano stubs (`--specs=nano.specs
--specs=nosys.specs`) and defines `PLATFORM_ARM`; the host build defines
`PLATFORM_HOST`.

---

## Regenerating the ML Model

`app/gen_model_data.py` builds the tiny Keras model, converts it to TFLite and
emits the C source (`model_data.h` / `model_data.cc`):

```bash
python app/gen_model_data.py
```

> Note: the output directory path near the top of the script is hard-coded —
> adjust `out_dir` to point at `app/` (or your target) before running.

To reproduce the DSP channelizer plots:

```bash
python app/channelizer.py
```

---

## Requirements

- A C11 / C++17 capable GCC toolchain (`gcc`, `g++`, `ar`, `size`).
- GNU `make`.
- For ARM builds: an `arm-none-eabi-` (or equivalent) cross toolchain.
- For the Python tooling: `numpy`, `matplotlib`, and `tensorflow`
  (only needed to regenerate the model / run the reference plots).

---

## License

Third-party components retain their original licenses:

- **PQClean** (`lib/crypto/pqclean`) — see the upstream PQClean license.
- **TensorFlow Lite for Microcontrollers** (`lib/ml/tflm`) — Apache 2.0.

The first-party DSP library, application glue and build system are provided as
part of this repository; add your project's license here.
