# ASS: Audio Serial Storage

**Save data to cassette** via FSK modulation, Reed–Solomon ECC, and LZMA compression.

A simple tape-style encoder/decoder originally inspired by 8‑bit microcomputers and the now‑deprecated [ctape project](https://github.com/windytan/ctape). Beyond cassette storage, ASS supports low‑bandwidth data links (e.g., amateur radio).

---

## Features

* **FSK Modulation/Demodulation** at 1200 Hz / 2400 Hz tones
* **LZMA Compression** (preset 9 + extreme) before transmission
* **Reed–Solomon (255, 223)** error correction (up to 16 byte-errors per block)
* **CRC32** payload integrity check
* **Live bit‑stream display** in console
* Read/write from WAV file or live audio input/output

---

## Prerequisites

* **Python** 3.7+

### Dependencies

Install via `pip`:

```bash
pip install numpy sounddevice reedsolo
```

---

## Installation

```bash
git clone https://github.com/snetting/ass.git
cd ass
chmod +x ass-reed.py
```

---

## Usage

### Encoding

* **File input**:

  ```bash
  ./ass-reed.py encode output.wav \
      --inputfile example.bin \
      --bitrate 2400 
  ```

* **Std‑in**:

  ```bash
  cat example.txt | \
      ./ass-reed.py encode output.wav \
      --bitrate 2400 
  ```

* **Inline string**:

  ```bash
  ./ass-reed.py encode output.wav \
      --data "Hello, world!" \
      --bitrate 2400 
  ```

### Decoding

* **From file**:

  ```bash
  ./ass-reed.py decode input.wav \
      --bitrate 2400 
  ```

* **Live via microphone**:

  ```bash
  ./ass-reed.py decode - \
      --bitrate 2400 
  ```

Decoded output is written to the original filename in the current directory.

---

## Inspiration

* My love of classic **8‑bit microcomputers** and their cassette storage
* The deprecated **ctape** project: [https://github.com/windytan/ctape](https://github.com/windytan/ctape)
* Low‑bandwidth digital modes in **amateur radio**

---

## Disclaimer

This code was partially generated and reviewed with the assistance of a large language model (LLM). It is provided as a work in progress; use at your own risk. Contributions, bug reports, and pull requests are welcome.

---

## To Do

* Add **MFSK encoding** to increase data rates
* Improve burst‑error handling with interleaving
* Implement proper sync and timing recovery (e.g. PLL / phase-locked loop) to adjust for clock drift and sample alignment

---

*Have fun sending bits the old-school way!*  
*73 de OH3SPN / M0SPN / TCM^SLP*

