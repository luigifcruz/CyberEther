#include "../generic.cc"


namespace Jetstream {

template<Device D, typename T>
struct FileWriter<D, T>::Impl {
};

template<Device D, typename T>
FileWriter<D, T>::FileWriter() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
FileWriter<D, T>::~FileWriter() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result FileWriter<D, T>::createCompute(const Context&) {
    JST_TRACE("Create File Writer compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::destroyCompute(const Context&) {
    JST_TRACE("Destroy File Writer compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::compute(const Context&) {
    if (config.recording) {
        gimpl->dataFile.write(reinterpret_cast<const char*>(input.buffer.data()), input.buffer.size_bytes());
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::GImpl::underlyingStartRecording() {
    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileWriter<D, T>::GImpl::underlyingStopRecording() {
    return Result::SUCCESS;
}

JST_FILE_WRITER_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
