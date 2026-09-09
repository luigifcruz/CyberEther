#ifndef JETSTREAM_DOMAINS_DSP_ADSB_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_ADSB_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Adsb : public Block::Config {
    JST_BLOCK_TYPE(adsb);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(XL);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "ADS-B Decoder",
        "Decodes ADS-B Mode S frames and maps aircraft positions.",
        "# ADS-B Decoder\n"
        "The ADS-B Decoder block takes raw CF32 IQ samples tuned to 1090 MHz "
        "at 2 MHz sample rate and decodes Mode S ADS-B transponder frames. "
        "It performs magnitude computation, preamble detection, bit decoding, "
        "CRC validation, and single-bit error correction using libmodes. "
        "Decoded aircraft positions are displayed on the existing world map "
        "with a STARS-inspired tracking overlay: compact targets, history dots, "
        "leader lines and persistent data blocks. This is an ADS-B visualization, "
        "not an FAA-certified air traffic control system.\n\n"

        "## Tracking Display\n"
        "The first data-block row is the callsign (or ICAO address). The second "
        "shows reported altitude in hundreds of feet and ground speed in tens "
        "of knots: 080 24 means 8,000 ft and 240 kt. Missing fields use dashes. "
        "Position reports older than 15 seconds are labeled STALE. Targets older "
        "than 120 seconds are hidden from the map.\n\n"

        "## Useful For\n"
        "- Aircraft tracking and identification.\n"
        "- ADS-B signal analysis and monitoring.\n"
        "- Aviation radio research.\n\n"

        "## Examples\n"
        "- Decode ADS-B from RTL-SDR:\n"
        "  Input: CF32[262144] (tuned to 1090 MHz, 2 MHz sample rate)\n"
        "  Output: Aircraft table with ICAO, callsign, position, altitude.\n\n"

        "## Implementation\n"
        "IQ Samples -> Magnitude -> Preamble Detection -> Bit Decode -> "
        "CRC -> Aircraft Table -> Map Display\n"
        "1. Convert CF32 IQ to magnitude vector.\n"
        "2. Detect Mode S preamble pattern.\n"
        "3. Decode and pack bits into message bytes.\n"
        "4. Validate CRC-24 with single-bit error correction.\n"
        "5. CPR decode aircraft positions.\n"
        "6. Display aircraft on interactive map.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_ADSB_BLOCK_HH
