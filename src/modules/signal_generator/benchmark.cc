#include "jetstream/modules/signal_generator.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename IT>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    // Small buffer
    JST_BENCHMARK_RUN("1024 samples", {
        .signalType = SignalType::Sine COMMA
        .sampleRate = 1000000.0 COMMA
        .frequency = 1000.0 COMMA
        .amplitude = 1.0 COMMA
        .bufferSize = 1024
    }, {}, IT);

    // Medium buffer
    JST_BENCHMARK_RUN("8192 samples", {
        .signalType = SignalType::Sine COMMA
        .sampleRate = 1000000.0 COMMA
        .frequency = 1000.0 COMMA
        .amplitude = 1.0 COMMA
        .bufferSize = 8192
    }, {}, IT);

    // Large buffer
    JST_BENCHMARK_RUN("65536 samples", {
        .signalType = SignalType::Sine COMMA
        .sampleRate = 1000000.0 COMMA
        .frequency = 1000.0 COMMA
        .amplitude = 1.0 COMMA
        .bufferSize = 65536
    }, {}, IT);

    // Complex waveforms
    JST_BENCHMARK_RUN("Chirp 8192", {
        .signalType = SignalType::Chirp COMMA
        .sampleRate = 1000000.0 COMMA
        .chirpStartFreq = 1000.0 COMMA
        .chirpEndFreq = 10000.0 COMMA
        .chirpDuration = 1.0 COMMA
        .amplitude = 1.0 COMMA
        .bufferSize = 8192
    }, {}, IT);

    // Noise generation
    JST_BENCHMARK_RUN("Noise 8192", {
        .signalType = SignalType::Noise COMMA
        .sampleRate = 1000000.0 COMMA
        .amplitude = 1.0 COMMA
        .noiseVariance = 1.0 COMMA
        .bufferSize = 8192
    }, {}, IT);
}

}  // namespace Jetstream