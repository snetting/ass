#!/usr/bin/env python3
import numpy as np
import wave
import struct
import argparse
import sys
import os
import sounddevice as sd
import zlib
import lzma
from reedsolo import RSCodec, ReedSolomonError

# Default config
SAMPLE_RATE = 44100
DEFAULT_BITRATE = 1200  # bits per second

# FSK frequencies
FREQ_0 = 1200
FREQ_1 = 2400

# Framing constants
PREAMBLE_BITS = [1, 0] * 32
SYNC_WORD = 0xAA55AA55
END_MARKER_WORD = 0x55AA55AA

# Reed–Solomon parameters (255, 223) code over GF(2^8)
RS_NSYM = 32              # number of parity symbols
RS_K = 255 - RS_NSYM      # number of data symbols per block
RS_CODEC = RSCodec(RS_NSYM)

# Compression preset for extreme xz-equivalent
LZMA_PRESET = 9 | lzma.PRESET_EXTREME

def rs_encode_blocks(data: bytes) -> bytes:
    """
    Split data into RS_K-sized blocks, encode each with RS, and concatenate.
    """
    out = bytearray()
    for i in range(0, len(data), RS_K):
        block = data[i:i+RS_K]
        codeword = RS_CODEC.encode(block)
        out.extend(codeword)
    return bytes(out)

def rs_decode_blocks(data: bytes):
    """
    Split received data into codewords, attempt RS decode, and recover original data.
    Returns (decoded_bytes, stats) where stats contains:
      - total_blocks
      - symbols_corrected
      - uncorrectable_blocks
    """
    out = bytearray()
    total_blocks = 0
    symbols_corrected = 0
    uncorrectable_blocks = 0
    cw_len = RS_K + RS_NSYM

    for i in range(0, len(data), cw_len):
        cw = data[i:i+cw_len]
        total_blocks += 1
        try:
            result = RS_CODEC.decode(cw)
            # result may be (decoded_bytes, errata_positions, …)
            if isinstance(result, tuple):
                decoded = result[0]
                # collect all errata positions from the tuple
                errata_positions = []
                for part in result[1:]:
                    if isinstance(part, (list, tuple)):
                        errata_positions.extend(part)
                symbols_corrected += len(errata_positions)
            else:
                decoded = result
            out.extend(decoded)
        except ReedSolomonError:
            uncorrectable_blocks += 1
            # on failure, strip off parity and take raw data
            out.extend(cw[:RS_K])

    stats = {
        "total_blocks": total_blocks,
        "symbols_corrected": symbols_corrected,
        "uncorrectable_blocks": uncorrectable_blocks,
    }
    return bytes(out), stats

def build_packet(filename: str, data: bytes) -> bytes:
    filename_bytes = filename.encode()
    crc = zlib.crc32(data)
    header = b'ASS1' + bytes([len(filename_bytes)]) + filename_bytes \
             + len(data).to_bytes(4, 'big') + crc.to_bytes(4, 'big')
    return header + data


def parse_packet(packet: bytes):
    if packet[:4] != b'ASS1':
        raise ValueError("Invalid packet format or missing magic header")
    name_len = packet[4]
    filename = packet[5:5 + name_len].decode()
    data_len = int.from_bytes(packet[5 + name_len:5 + name_len + 4], 'big')
    crc = int.from_bytes(packet[5 + name_len + 4:5 + name_len + 8], 'big')
    data = packet[5 + name_len + 8:5 + name_len + 8 + data_len]
    return filename, data, crc


def generate_tone(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    return (32767 * tone).astype(np.int16)


def encode_bits(bits, bit_duration, sample_rate):
    signal = []
    for bit in bits:
        freq = FREQ_1 if bit else FREQ_0
        tone = generate_tone(freq, bit_duration, sample_rate)
        signal.extend(tone)
    return np.array(signal, dtype=np.int16)


def encode(data: bytes, outfile: str, bitrate: int):
    # data here is your full packet (header+compressed payload)
    print(f"[ENCODE] Packet size: {len(data)} bytes")
    # Apply Reed–Solomon encoding
    data_rs = rs_encode_blocks(data)
    overhead = len(data_rs) - len(data)
    print(f"[ENCODE] RS-encode: {len(data)} → {len(data_rs)} bytes (+{overhead} bytes parity)")

    bit_duration = 1.0 / bitrate
    bits = PREAMBLE_BITS.copy()

    # Sync word
    SYNC_BITS = [(SYNC_WORD >> (31 - i)) & 1 for i in range(32)]
    bits += SYNC_BITS

    # Payload bits
    for byte in data_rs:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)

    # End markers (×3)
    END_BITS = [(END_MARKER_WORD >> (31 - i)) & 1 for i in range(32)]
    bits += END_BITS * 3

    # Silence
    bits += [0] * 32

    signal = encode_bits(bits, bit_duration, SAMPLE_RATE)

    if outfile == '-':
        print("Playing audio live...")
        sd.play(signal / 32767.0, SAMPLE_RATE)
        sd.wait()
    else:
        with wave.open(outfile, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(signal.tobytes())
        print(f"Data encoded to {outfile}")


def detect_bits(signal, bit_duration, sample_rate):
    samples_per_bit = int(sample_rate * bit_duration)
    bits = []
    for i in range(0, len(signal), samples_per_bit):
        chunk = signal[i:i + samples_per_bit]
        if len(chunk) < samples_per_bit:
            break
        fft = np.abs(np.fft.fft(chunk))
        freqs = np.fft.fftfreq(len(chunk), 1 / sample_rate)
        peak = abs(freqs[np.argmax(fft[:len(fft)//2])])
        bits.append(1 if abs(peak - FREQ_1) < abs(peak - FREQ_0) else 0)
    return bits


def find_sync(bits):
    sync_bits = [(SYNC_WORD >> (31 - i)) & 1 for i in range(32)]
    for i in range(len(bits) - len(sync_bits)):
        if bits[i:i + len(sync_bits)] == sync_bits:
            print("[DECODE] Found sync pattern.")
            return i + len(sync_bits)
    return None


def bits_to_bytes(bits):
    data = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | b
        data.append(byte)
    return data


def decode_wav(filename, bitrate):
    print(f"[DECODE] Loading WAV: {filename} @ {bitrate}bps")
    with wave.open(filename, 'r') as wf:
        frames = wf.readframes(wf.getnframes())
        signal = np.frombuffer(frames, dtype=np.int16)
    decode_signal(signal, bitrate)


def record_and_decode(bitrate):
    import sounddevice as sd
    import numpy as np
    import wave
    import sys

    # Listen indefinitely until Ctrl+C or end marker
    block = int(SAMPLE_RATE * (1/bitrate) * 10)
    rec_bits = []
    buf = []
    line_buffer = []

    end_bits = [(END_MARKER_WORD >> (31 - i)) & 1 for i in range(32)]
    try:
        print("[DECODE] Listening... Ctrl+C to stop when ready.")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
            while True:
                block_data, _ = stream.read(block)
                sig = (block_data[:,0] * 32767).astype(np.int16)
                buf.append(sig)

                bits = detect_bits(sig, 1/bitrate, SAMPLE_RATE)
                for bit in bits:
                    rec_bits.append(bit)
                    line_buffer.append('*' if bit else '.')
                    sys.stdout.write('\r' + ''.join(line_buffer[-80:]))
                    sys.stdout.flush()

                    if len(rec_bits) >= 32 and rec_bits[-32:] == end_bits:
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\n[DECODE] Stopped listening.")

    full = np.concatenate(buf)
    #with wave.open('debug.wav','w') as wf:
    #    wf.setnchannels(1)
    #    wf.setsampwidth(2)
    #    wf.setframerate(SAMPLE_RATE)
    #    wf.writeframes(full.tobytes())
    #print("[DECODE] Saved debug.wav")
    print("[DECODE] Decoding...")
    decode_signal(full, bitrate)


def decode_signal(signal, bitrate):
    print(f"[DECODE] Decoding {len(signal)} samples @ {bitrate}bps")
    bits = detect_bits(signal, 1/bitrate, SAMPLE_RATE)
    print(f"[DECODE] Got {len(bits)} bits.")

    idx = find_sync(bits)
    if idx is None:
        print("[DECODE] No sync; abort")
        return
    data_bits = bits[idx:]
    end_idx = find_end_marker(data_bits)
    if end_idx is not None:
        data_bits = data_bits[:end_idx]
    else:
        print("[DECODE] No end marker; using full stream.")

    raw = bits_to_bytes(data_bits)
    print("[DECODE] Applying RS decode...")
    decoded, rs_stats = rs_decode_blocks(raw)

    print(f"[DECODE] RS blocks: {rs_stats['total_blocks']}, "
          f"symbols corrected: {rs_stats['symbols_corrected']}, "
          f"uncorrectable: {rs_stats['uncorrectable_blocks']}")

    try:
        fname, payload, crc = parse_packet(decoded)
        print(f"[DECODE] Filename: {fname}")
        if zlib.crc32(payload) != crc:
            print(f"[DECODE] CRC mismatch: expected {crc}")
        else:
            print("[DECODE] CRC OK")
            # decompress
            try:
                outdata = lzma.decompress(payload)
                print("[DECODE] Decompression successful.")
            except Exception as e:
                print(f"[DECODE] Decompression failed: {e}")
                outdata = payload
            with open(fname,'wb') as f:
                f.write(outdata)
            print(f"[DECODE] Wrote output file: {fname}")
    except Exception as e:
        print(f"[DECODE] Failed parse/write: {e}")


def find_end_marker(bits):
    end_bits = [(END_MARKER_WORD >> (31 - i)) & 1 for i in range(32)]
    for i in range(len(bits)-len(end_bits)):
        if bits[i:i+len(end_bits)]==end_bits:
            print("[DECODE] Found end marker.")
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description="FSK with RS ECC + LZMA compression")
    parser.add_argument('mode',choices=['encode','decode'])
    parser.add_argument('file',help="outfile for encode; infile or '-' for live decode")
    parser.add_argument('--data',help="string to encode")
    parser.add_argument('--inputfile',help="file to encode")
    parser.add_argument('--bitrate',type=int,default=DEFAULT_BITRATE)
    if len(sys.argv)==1:
        parser.print_help(); sys.exit(1)
    args = parser.parse_args()

    if args.mode=='encode':
        # gather raw data
        if args.data:
            raw = args.data.encode(); name = "inline_data.txt"
        elif args.inputfile:
            raw = open(args.inputfile,'rb').read(); name = os.path.basename(args.inputfile)
        else:
            raw = sys.stdin.buffer.read(); name = "stdin_input.bin"
        print(f"[MAIN] Read      : {len(raw)} bytes from '{name}'")

        # compress
        comp = lzma.compress(raw, preset=LZMA_PRESET)
        saved = len(raw) - len(comp)
        pct   = 100.0 * (saved / len(raw)) if raw else 0
        print(f"[MAIN] Compressed: {len(comp)} bytes (saved {saved} bytes, {pct:.1f}% reduction)")

        # 3) CRC
        crc = zlib.crc32(comp)
        print(f"[MAIN] CRC32     : 0x{crc:08X}")

        packet = build_packet(name, comp)
        print(f"[MAIN] Packet    : {len(packet)} bytes (hdr+name+len+crc+data)")
        encode(packet, args.file, args.bitrate)

    else:
        print("[MAIN] Entering decode mode")
        if args.file=='-':
            record_and_decode(args.bitrate)
        else:
            decode_wav(args.file, args.bitrate)

if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
