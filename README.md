# ass
Audio Serial Storage (ASS) - Save data to cassette

A simple FSK tape-style encoder/decoder with Reed–Solomon error correction and LZMA compression. Originally inspired by 8‑bit microcomputers and the now‑deprecated ctape project. In addition to storing data on cassette tape, ASS can be used for low‑bandwidth data communications (e.g. amateur radio).

#Features

FSK Modulation/Demodulation at 1200 Hz / 2400 Hz tones

LZMA Compression (preset 9 + extreme) before transmission

Reed–Solomon (255, 223) error correction (up to 16 byte errors/block)

CRC32 validation of payload integrity

Live bit‑stream display in console

Read/write from WAV file or live microphone input

#Prerequisites

Python 3.7+

#Dependencies:

pip install numpy sounddevice reedsolo lzma-wave

(lzma is part of the standard library; lzma-wave is a placeholder if needed for wave support.)

#Installation

Clone this repository and make the script executable:

git clone https://your.repo.url/ass.git
cd ass
chmod +x ass-reed.py

#Usage

Encoding

File input:

./ass-reed.py encode output.wav --inputfile example.bin --bitrate 100

Std‑in:

cat example.txt | ./ass-reed.py encode output.wav --bitrate 100

Inline string:

./ass-reed.py encode output.wav --data "Hello, world!" --bitrate 100

Decoding

From file:

./ass-reed.py decode input.wav --bitrate 100

Live via microphone:

./ass-reed.py decode - --bitrate 100

After decoding, the recovered file will be saved in the current directory.

#Inspiration

My love of classic 8‑bit microcomputers and their tape storage

The deprecated ctape project: https://github.com/windytan/ctape

Low‑bandwidth digital modes in amateur radio

#Disclaimer

This code was partially generated and reviewed with the assistance of a large language model (LLM). It is provided as a work in progress; use at your own risk. Contributions, bug reports, and pull requests are welcome.

#To Do

Add MFSK encoding to increase achievable data rates

Improve burst‑error handling with interleaving




