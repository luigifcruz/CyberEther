#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

std::string to_timestamp(int64_t t) {
    int64_t sec = t/100;
    int64_t msec = t - sec*100;
    int64_t min = sec/60;
    sec = sec - min*60;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int) min, (int) sec, (int) msec);

    return std::string(buf);
}

namespace Jetstream {

template<Device D, typename T>
Result SpeechRecognition<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Speech Recognition compute core using CPU backend.");

    ctx = whisper_init_from_file("/Users/luigi/sandbox/whisper.cpp/models/ggml-base.bin");
    wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.translate        = false;
    wparams.single_segment   = false;
    wparams.max_tokens       = 128;
    wparams.language         = "pt";
    wparams.n_threads        = 8;
    wparams.speed_up         = false;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result SpeechRecognition<D, T>::compute(const RuntimeMetadata&) {
    wparams.prompt_tokens    = promptTokens.data();
    wparams.prompt_n_tokens  = promptTokens.size();

    if (whisper_full(ctx, wparams, input.buffer.data(), input.buffer.size()) != 0) {
        JST_FATAL("failed to process audio\n");
    }

    const int n_segments = whisper_full_n_segments(ctx);

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);

        std::string output = text;
        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
            output += " [SPEAKER_TURN]";
        }

        output += "\n";

        JST_TRACE(">> {}", output);

        textBuffer += output.c_str();
    }

    promptTokens.clear();
    for (int i = 0; i < n_segments; ++i) {
        const int token_count = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < token_count; ++j) {
            promptTokens.push_back(whisper_full_get_token_id(ctx, i, j));
        }
    }

    return Result::SUCCESS;
}

JST_SPEECH_RECOGNITION_CPU(JST_INSTANTIATION);

}  // namespace Jetstream
