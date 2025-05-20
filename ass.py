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
import time
import datetime
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

def build_packet(filename: str, data: bytes, flags: int) -> bytes:
    """
    Build the packet header, now honours the `flags` argument.
    """
    filename_bytes = filename.encode()
    crc = zlib.crc32(data)

    # Gather file metadata
    try:
        st = os.stat(filename)
        backup_ts = int(time.time())
        ctime     = int(st.st_ctime)
        mtime     = int(st.st_mtime)
        uid       = st.st_uid
        gid       = st.st_gid
        mode      = st.st_mode & 0o777
    except Exception:
        backup_ts = ctime = mtime = uid = gid = mode = 0

    header = (
        b'ASS1'
        + bytes([len(filename_bytes)])
        + filename_bytes
        + backup_ts.to_bytes(4, 'big')
        + ctime.to_bytes(4, 'big')
        + mtime.to_bytes(4, 'big')
        + uid.to_bytes(4, 'big')
        + gid.to_bytes(4, 'big')
        + mode.to_bytes(2, 'big')
        + bytes([flags])                      # <<-- use the flags you passed
        + len(data).to_bytes(4, 'big')
        + crc.to_bytes(4, 'big')
    )
    return header + data

def parse_packet(packet: bytes):
    """
    Extract:
      filename, data_bytes, crc,
      backup_ts, ctime, mtime,
      uid, gid, mode, flags
    """
    if packet[:4] != b'ASS1':
        raise ValueError("Invalid packet format or missing magic header")

    idx = 4
    name_len = packet[idx]
    idx += 1

    filename = packet[idx:idx + name_len].decode()
    idx += name_len

    backup_ts = int.from_bytes(packet[idx:idx+4], 'big'); idx += 4
    ctime     = int.from_bytes(packet[idx:idx+4], 'big'); idx += 4
    mtime     = int.from_bytes(packet[idx:idx+4], 'big'); idx += 4
    uid       = int.from_bytes(packet[idx:idx+4], 'big'); idx += 4
    gid       = int.from_bytes(packet[idx:idx+4], 'big'); idx += 4
    mode      = int.from_bytes(packet[idx:idx+2], 'big'); idx += 2
    flags     = packet[idx]; idx += 1

    data_len = int.from_bytes(packet[idx:idx+4], 'big'); idx += 4
    crc      = int.from_bytes(packet[idx:idx+4], 'big'); idx += 4

    data = packet[idx:idx + data_len]
    return filename, data, crc, backup_ts, ctime, mtime, uid, gid, mode, flags

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
    """
    Demodulate bits → bytes → Reed-Solomon decode → parse header → restore metadata → write file
    """
    print(f"[DECODE] Decoding {len(signal)} samples @ {bitrate}bps")
    # 1) Demodulate to raw bits
    bits = detect_bits(signal, 1/bitrate, SAMPLE_RATE)
    print(f"[DECODE] Got {len(bits)} bits.")

    # 2) Find sync and trim
    start_idx = find_sync(bits)
    if start_idx is None:
        print("[DECODE] No sync; abort")
        return
    data_bits = bits[start_idx:]

    # 3) Find end marker and trim
    end_idx = find_end_marker(data_bits)
    if end_idx is not None:
        data_bits = data_bits[:end_idx]
    else:
        print("[DECODE] No end marker; using full stream.")

    # 4) Pack bits into bytes
    raw = bits_to_bytes(data_bits)

    # 5) Reed–Solomon decode with stats
    print("[DECODE] Applying RS decode...")
    decoded, rs_stats = rs_decode_blocks(raw)
    print(
        f"[DECODE] RS blocks: {rs_stats['total_blocks']}, "
        f"symbols corrected: {rs_stats['symbols_corrected']}, "
        f"uncorrectable: {rs_stats['uncorrectable_blocks']}"
    )

    # 6) Parse header & metadata
    try:
        (
            fname,
            payload,
            crc,
            backup_ts,
            ctime,
            mtime,
            uid,
            gid,
            mode,
            flags,
        ) = parse_packet(decoded)
    except Exception as e:
        print(f"[DECODE] Failed to parse header: {e}")
        return

    # 7) Show header info
    print(f"[DECODE] Filename:         {fname}")
    print(f"[DECODE] Backup timestamp: {datetime.datetime.fromtimestamp(backup_ts)}")
    print(f"[DECODE] Original ctime:   {datetime.datetime.fromtimestamp(ctime)}")
    print(f"[DECODE] Original mtime:   {datetime.datetime.fromtimestamp(mtime)}")
    print(f"[DECODE] UID: {uid}, GID: {gid}, Mode: {oct(mode)}, Flags: {flags}")

    # NEW: report compression flag
    if flags & 1:
        print("[DECODE] Compression flag: SET  (payload was compressed at encode time)")
    else:
        print("[DECODE] Compression flag: NOT SET")

    # 8) Decompress payload if flag bit0 set
    if flags & 1:
        try:
            payload = lzma.decompress(payload)
            print("[DECODE] Decompression successful.")
        except Exception as e:
            print(f"[DECODE] Decompression failed: {e}")

    # 9) Restore ownership, permissions, and timestamps
    try:
        os.chown(fname, uid, gid)
        os.chmod(fname, mode)
        os.utime(fname, (mtime, mtime))
        print("[DECODE] Restored ownership and timestamps.")
    except PermissionError:
        print("[DECODE] Warning: insufficient privileges to restore owner/group.")
    except Exception as e:
        print(f"[DECODE] Metadata restore failed: {e}")

    # 10) Write file to disk
    try:
        with open(fname, 'wb') as f:
            f.write(payload)
        print(f"[DECODE] Wrote output file: {fname}")
    except Exception as e:
        print(f"[DECODE] Failed to write output file: {e}")


def find_end_marker(bits):
    end_bits = [(END_MARKER_WORD >> (31 - i)) & 1 for i in range(32)]
    for i in range(len(bits)-len(end_bits)):
        if bits[i:i+len(end_bits)]==end_bits:
            print("[DECODE] Found end marker.")
            return i
    return None


def main():
    parser = argparse.ArgumentParser(
        prog='ass.py',
        description="FSK with RS ECC + LZMA compression",
        usage=(
            "ass.py [-h] {encode,decode} "
            "[--data DATA] [--inputfile INPUTFILE] "
            "[--bitrate BITRATE] "
            "[--alwayscompress | --nocompress | --autocompress] "
            "file"
        )
    )

    # positional mode + file
    parser.add_argument(
        'mode',
        choices=['encode', 'decode'],
        help="Mode: 'encode' to audio, 'decode' to extract"
    )
    parser.add_argument(
        'file',
        help="Output WAV file for encode, or input WAV (or '-' for live input) for decode"
    )

    # optional data sources
    parser.add_argument('--data', help="Inline string to encode")
    parser.add_argument('--inputfile', help="Path to input file to encode")

    # bitrate with default
    parser.add_argument(
        '--bitrate',
        type=int,
        default=DEFAULT_BITRATE,
        help=f"Bitrate in bits/sec (default: {DEFAULT_BITRATE})"
    )

    # mutually exclusive compression flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--alwayscompress', '--always-compress',
        action='store_true',
        dest='always_compress',
        help="Always apply LZMA compression"
    )
    group.add_argument(
        '--nocompress', '--no-compress',
        action='store_true',
        dest='no_compress',
        help="Disable compression entirely"
    )
    group.add_argument(
        '--autocompress', '--auto-compress',
        action='store_true',
        dest='auto_compress',
        help="Compress only if it makes the data smaller"
    )
    parser.set_defaults(auto_compress=True)

    if len(sys.argv)==1:
        parser.print_help(); sys.exit(1)
    args = parser.parse_args()

    if args.mode == 'encode':
        # ─── 1) Read raw data ──────────────────────────────────────────────
        if args.data:
            raw = args.data.encode()
            name = "inline_data.txt"
        elif args.inputfile:
            raw = open(args.inputfile, 'rb').read()
            name = os.path.basename(args.inputfile)
        else:
            raw = sys.stdin.buffer.read()
            name = "stdin_input.bin"
        print(f"[ENCODE] Read data: {len(raw)} bytes from '{name}'")

        # 2) Decide on compression
        if args.always_compress:
            payload = lzma.compress(raw, preset=LZMA_PRESET)
            flags = 1
            print(f"[ENCODE] Compression: forced, result {len(payload)} bytes")
        elif args.no_compress:
            payload = raw
            flags = 0
            print(f"[ENCODE] Compression: disabled, sending {len(payload)} bytes")
        else:  # autocompress
            comp = lzma.compress(raw, preset=LZMA_PRESET)
            if len(comp) < len(raw):
                payload = comp; flags = 1
                saved = len(raw) - len(comp)
                print(f"[ENCODE] Compression: applied, {len(comp)} bytes ({saved} bytes saved)")
            else:
                payload = raw; flags = 0
                print(f"[ENCODE] Compression: skipped (compressed size {len(comp)} ≥ raw size {len(raw)})")

        # ─── 2) Gather file metadata ─────────────────────────────────────
        try:
            st = os.stat(name)
            backup_ts = int(time.time())
            ctime     = int(st.st_ctime)
            mtime     = int(st.st_mtime)
            uid       = st.st_uid
            gid       = st.st_gid
            mode_perm = st.st_mode & 0o777
        except FileNotFoundError:
            backup_ts = ctime = mtime = uid = gid = mode_perm = 0

        print("[ENCODE] Metadata:")
        print(f"    Backup timestamp: {datetime.datetime.fromtimestamp(backup_ts)}")
        print(f"    Original ctime:   {datetime.datetime.fromtimestamp(ctime)}")
        print(f"    Original mtime:   {datetime.datetime.fromtimestamp(mtime)}")
        print(f"    UID: {uid}, GID: {gid}, Mode: {oct(mode_perm)}, Flags: {flags}")


        # ─── 4) CRC32 ──────────────────────────────────────────────────────
        crc = zlib.crc32(payload)
        print(f"[ENCODE] CRC32: 0x{crc:08X}")

        # ─── 5) Packetize ─────────────────────────────────────────────────
        packet = build_packet(name, payload, flags)
        print(f"[ENCODE] Packet: {len(packet)} bytes (hdr+data+crc)")

        # ─── 6) Modulate & write ──────────────────────────────────────────
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
