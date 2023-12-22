#include "jetstream/modules/speech_recognition.hh"

namespace Jetstream {

template<Device D, typename T>
Result SpeechRecognition<D, T>::create() {
    JST_DEBUG("Initializing Speech Recognition module.");
    JST_INIT_IO();

    return Result::SUCCESS;
}

template<Device D, typename T>
void SpeechRecognition<D, T>::info() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
