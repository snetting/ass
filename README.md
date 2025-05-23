ASS: Audio Serial Storage
=========================

**Save data to cassette** via FSK modulation, Reed--Solomon ECC, and LZMA compression.

A simple tape‑style encoder/decoder originally inspired by 8‑bit microcomputers and the now‑deprecated ctape project. Beyond cassette storage, ASS supports low‑bandwidth data links (e.g., amateur radio).

* * * * *

Features
--------

-   **FSK Modulation/Demodulation** at 1200 Hz / 2400 Hz tones

-   **LZMA Compression** (xz‑9e) with three modes: `--alwayscompress`, `--nocompress`, or `--autocompress` (default)

-   **Reed--Solomon (255, 223)** error correction (up to 16 byte‑errors per block)

-   **CRC32** payload integrity check

-   **Live bit‑stream display** in console

-   **Read/write from WAV file or live audio input**

-   **Preserve and restore file metadata:** timestamps, ownership, and permissions

-   **Experimental** MFSK 2,4,8,16 support
* * * * *

Serial Data Structure
---------------------

Each packet on the wire is arranged as follows (all multi‑byte fields big‑endian):

| Offset | Length | Field                                 |
|:------:|:------:|:--------------------------------------|
| 0      | 4      | Magic header: `"ASS1"`                |
| 4      | 1      | Filename length (N)                   |
| 5      | N      | Filename (UTF-8)                      |
| 5+N    | 4      | Backup timestamp (UNIX epoch seconds) |
| 9+N    | 4      | Original ctime                        |
| 13+N   | 4      | Original mtime                        |
| 17+N   | 4      | UID                                   |
| 21+N   | 4      | GID                                   |
| 25+N   | 2      | File mode (permissions)               |
| 27+N   | 1      | Flags (bit 0 = compression used)      |
| 28+N   | 4      | Payload byte-length                   |
| 32+N   | 4      | CRC32 of payload                      |
| 36+N   | M      | Payload (compressed or raw data)      |

After framing, each byte goes through Reed--Solomon → FSK → WAV.

* * * * *

Prerequisites
-------------

-   **Python** 3.7+

### Dependencies

Install via `pip`:

```
pip install numpy sounddevice reedsolo
```

> `*lzma*`* is part of Python's standard library.*

* * * * *

Installation
------------

```
git clone https://your.repo.url/ass.git
cd ass
chmod +x ass.py
```

* * * * *

Usage
-----

```
usage: ass.py [-h] {encode,decode} [--data DATA] [--inputfile INPUTFILE]
             [--bitrate BITRATE] [--mfsk {2,4,8,16}]
             [--alwayscompress | --nocompress | --autocompress] 
             [--noclamp] file
```

-   **Positional arguments**:

    -   `encode, decode` : Mode of operation

    -   `file` : Output WAV when encoding; input WAV (or `-` for live audio) when decoding

-   **Optional arguments**:

    -   `-h, --help` : Show this help message and exit

    -   `--data DATA` : Inline string to encode

    -   `--inputfile FILE` : Path to input file to encode

    -   `--bitrate BITRATE` : Bitrate in bits/sec (default: 1200)
    
    -   `--mfsk {2,4,8,16}` : Number of tones (2 = standard FSK; 4/8/16 = experimental MFSK)

    -   `--noclamp` : bypass bitrate clamping (use with care!

    -   **Compression modes** (mutually exclusive):

        -   `--alwayscompress` : Always apply LZMA

        -   `--nocompress` : Disable LZMA

        -   `--autocompress` : Apply only if compressed < raw (default)

### Examples

```
# Encode with auto‑compress (default)
./ass.py encode output.wav --inputfile example.bin --bitrate 2400

# Encode forcing compression
./ass.py encode output.wav --inputfile example.bin --bitrate 2400 --alwayscompress

# Encode without compression
./ass.py encode output.wav --inputfile example.bin --bitrate 2400 --nocompress

# Decode from file
./ass.py decode input.wav --bitrate 2400 

# Decode live from mic
./ass.py decode - --bitrate 2400 

# Experimental 8-FSK at 100 bps bit-rate (effectively 300 bps)
./ass.py encode out-mfsk.wav --inputfile data.bin --bitrate 100 --mfsk 8

# Decode a 16-FSK transmission
./ass.py decode in-mfsk16.wav --bitrate 1200 --mfsk 16

```

# Bitrate Clamping

To ensure reliable tone discrimination, ASS by default limits your requested `--bitrate` so that the FFT’s frequency resolution (≈ bitrate Hz) does not exceed the spacing between adjacent M-FSK tones:

```
tone_spacing = (FREQ_1 - FREQ_0) / (M - 1)
max_bitrate  = floor(tone_spacing)
```

- **2-FSK (M=2)**: tone_spacing = 1200 Hz → max_bitrate = 1200 bps  
- **16-FSK (M=16)**: tone_spacing ≈ 1200 Hz / 15 ≈ 80 Hz → max_bitrate = 80 bps → effective bit-rate = 80 × 4 = 320 bps  

If you request a higher bitrate, ASS prints:

```
[MAIN] Warning: 8-FSK at 300 bps gives Δf_res≈300 Hz > tone spacing 171.4 Hz;
    → clamping to 171 bps
```

Add `--noclamp` to bypass the clamp (you’ll still see the warning, but ASS will respect your requested bitrate).

Decoded files are written with their original names, metadata, ownership, and permissions.

* * * * *

Inspiration
-----------

-   My love of classic **8‑bit microcomputers** and cassette storage

-   The deprecated **ctape** project: https://github.com/windytan/ctape

-   Low‑bandwidth digital modes in **amateur radio**

* * * * *

Disclaimer
----------

This code was partially generated and reviewed with the assistance of a large language model (LLM). It is provided as a work in progress; use at your own risk. Contributions and pull requests are welcome.

* * * * *

To Do
-----

-   Add **MFSK encoding** to increase data rates (in progress / Experimental)

-   Improve burst‑error handling with interleaving

-   Implement proper sync and timing recovery (e.g., PLL)

-   (Optional) GUI

* * * * *

*Have fun sending bits the old‑school way!*\
*73 de OH3SPN / M0SPN / TCM^SLP*
