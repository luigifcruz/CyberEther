#include "sweep_decoder.hh"

#include <algorithm>

#include "../../tools/samples.hh"

namespace Jetstream::Radio {

std::optional<HackRfSweepDecoder::Block> HackRfSweepDecoder::Block::Parse(const U8* data, const U64 size,
                                                              const U64 blockBytes) {
    if (!data || blockBytes <= HeaderBytes || size < blockBytes ||
        data[0] != 0x7f || data[1] != 0x7f) {
        return std::nullopt;
    }

    U64 frequency = 0;
    for (U64 i = 0; i < 8; ++i) {
        frequency |= static_cast<U64>(data[2 + i]) << (8 * i);
    }

    return Block{
        .frequency = frequency,
        .payload = reinterpret_cast<const I8*>(data + HeaderBytes),
        .payloadSamples = (blockBytes - HeaderBytes) / 2,
    };
}

void HackRfSweepDecoder::Sequence::configure(std::vector<U64> nextFrequencies, const U64 nextDwell) {
    frequencies = std::move(nextFrequencies);
    dwell = std::max<U64>(nextDwell, 1);
    invalidate();
    lost = 0;
}

void HackRfSweepDecoder::Sequence::invalidate() {
    if (synchronized) {
        ++lost;
    }
    synchronized = false;
    seenOther = false;
    hop = 0;
    repetition = 0;
}

HackRfSweepDecoder::Sequence::Event HackRfSweepDecoder::Sequence::next(const U64 frequency, U64& captureHop,
                                                           U64& captureRepetition) {
    if (frequencies.empty()) {
        return Event::Invalid;
    }
    if (std::find(frequencies.begin(), frequencies.end(), frequency) == frequencies.end()) {
        invalidate();
        return Event::Invalid;
    }

    if (!synchronized) {
        if (frequency != frequencies.front()) {
            seenOther = true;
            return Event::Waiting;
        }
        if (!seenOther) {
            return Event::Waiting;
        }
        synchronized = true;
        hop = 0;
        repetition = 0;
    }

    if (frequency != frequencies[hop]) {
        const bool first = frequency == frequencies.front();
        invalidate();
        seenOther = !first;
        return Event::Invalid;
    }

    captureHop = hop;
    captureRepetition = repetition;
    if (++repetition == dwell) {
        repetition = 0;
        hop = (hop + 1) % frequencies.size();
    }
    return Event::Capture;
}

U64 HackRfSweepDecoder::Sequence::losses() const {
    return lost;
}

void HackRfSweepDecoder::configure(std::vector<U64> headers, const U64 nextDwell,
                             const U64 nextCaptureLength, const U64 nextBlockBytes,
                             const F64 nextOffset) {
    dwell = std::max<U64>(nextDwell, 1);
    captureLength = nextCaptureLength;
    blockBytes = nextBlockBytes;
    offset = nextOffset;
    captures.assign(captureLength * dwell, CF32{});
    valid = 0;
    sequence.configure(std::move(headers), dwell);
}

bool HackRfSweepDecoder::decode(const HackRfTransferQueue::Transfer& transfer,
                          const Device::SweepSink& sink) {
    if (transfer.gapBefore) {
        sequence.invalidate();
    }

    const U64 complete = blockBytes > 0 ? transfer.bytes.size() / blockBytes : 0;
    for (U64 b = 0; b < complete; ++b) {
        const auto block = Block::Parse(transfer.bytes.data() + b * blockBytes, blockBytes,
                                        blockBytes);
        if (!block || block->payloadSamples < captureLength) {
            sequence.invalidate();
            continue;
        }
        U64 hop = 0;
        U64 repetition = 0;
        const auto event = sequence.next(block->frequency, hop, repetition);
        if (event == Sequence::Event::Invalid) {
            continue;
        }
        ++valid;
        if (event != Sequence::Event::Capture) {
            continue;
        }
        const I8* payload = block->payload + (block->payloadSamples - captureLength) * 2;
        ConvertI8(payload, captures.data() + repetition * captureLength, captureLength);
        if (repetition + 1 < dwell) {
            continue;
        }
        const Device::SweepCapture capture{
            .hop = hop,
            .frequency = static_cast<F64>(block->frequency) + offset,
            .samples = captures,
        };
        if (!sink(capture)) {
            return false;
        }
    }

    if (blockBytes == 0 || transfer.bytes.size() % blockBytes != 0) {
        sequence.invalidate();
    }
    return true;
}

U64 HackRfSweepDecoder::blocks() const {
    return valid;
}

U64 HackRfSweepDecoder::losses() const {
    return sequence.losses();
}

}  // namespace Jetstream::Radio
