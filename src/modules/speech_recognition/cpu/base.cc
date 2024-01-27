#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

std::string to_timestamp(int64_t t, bool comma = false) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

namespace Jetstream {

template<Device D, typename T>
Result SpeechRecognition<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Speech Recognition compute core using CPU backend.");

    struct whisper_context_params cparams;
    cparams.use_gpu = true;
    ctx = whisper_init_from_file_with_params("/Users/luigi/sandbox/whisper.cpp/models/ggml-base.bin", cparams);
    wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

    wparams.print_progress   = false;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.translate        = false;
    wparams.single_segment   = false;
    wparams.language         = "en";
    wparams.n_threads        = 8;
    wparams.speed_up         = false;

    wparams.token_timestamps = true;
    wparams.max_len          = 1;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result SpeechRecognition<D, T>::compute(const Context&) {
    if (whisper_full(ctx, wparams, input.buffer.data(), input.buffer.size()) != 0) {
        JST_FATAL("failed to process audio\n");
    }

    const int n_segments = whisper_full_n_segments(ctx);

    for (int i = 0; i < n_segments; ++i) {
        const auto t0 = whisper_full_get_segment_t0(ctx, i);
        const auto t1 = whisper_full_get_segment_t1(ctx, i);

        const char* text = whisper_full_get_segment_text(ctx, i);

        std::string output = text;
        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
            output += " [SPEAKER_TURN]";
        }

        JST_TRACE(">> [{} -> {}] {}", to_timestamp(t0), to_timestamp(t1), output);

        textBuffer += text;
    }

    textBuffer += "\n";

    return Result::SUCCESS;
}

JST_SPEECH_RECOGNITION_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
