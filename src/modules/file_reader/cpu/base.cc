#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct FileReader<D, T>::Impl {
};

template<Device D, typename T>
FileReader<D, T>::FileReader() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
FileReader<D, T>::~FileReader() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result FileReader<D, T>::createCompute(const Context&) {
    JST_TRACE("Create File Reader compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileReader<D, T>::destroyCompute(const Context&) {
    JST_TRACE("Destroy File Reader compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileReader<D, T>::compute(const Context&) {
    if (config.playing) {
        const U64 bytesToRead = output.buffer.size_bytes();
        const U64 remainingBytes = gimpl->fileSize - gimpl->currentPosition;

        if (remainingBytes == 0) {
            if (config.loop) {
                gimpl->dataFile.seekg(0, std::ios::beg);
                gimpl->currentPosition = 0;
            } else {
                return Result::YIELD;
            }
        }

        const U64 actualBytesToRead = std::min(bytesToRead, remainingBytes);

        // Read data from file
        gimpl->dataFile.read(reinterpret_cast<char*>(output.buffer.data()), actualBytesToRead);
        gimpl->currentPosition += gimpl->dataFile.gcount();
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileReader<D, T>::GImpl::underlyingStartPlaying() {
    return Result::SUCCESS;
}

template<Device D, typename T>
Result FileReader<D, T>::GImpl::underlyingStopPlaying() {
    return Result::SUCCESS;
}

JST_FILE_READER_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
