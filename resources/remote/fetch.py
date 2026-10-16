#!/bin/python3

import gzip
import sys
import urllib.request


def fetch(url, output, symbol):
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CyberEther-Build"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        gzip.decompress(data)
    except Exception as error:
        sys.exit(f"Failed to download the remote viewer page from {url}: {error}")

    with open(output, "w", encoding="utf-8") as f:
        f.write("#include <cstddef>\n\n")
        f.write("namespace Jetstream::Resources {\n\n")
        f.write(f"extern const unsigned char {symbol}[] = {{\n")
        for offset in range(0, len(data), 16):
            chunk = data[offset:offset + 16]
            f.write("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",\n")
        f.write("};\n\n")
        f.write(f"extern const std::size_t {symbol}Size = {len(data)};\n\n")
        f.write("}  // namespace Jetstream::Resources\n")


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2], sys.argv[3])
