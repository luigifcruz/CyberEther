#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "jetstream/fft/base.hpp"
#include "jetstream/histogram/base.hpp"
#include "jetstream/lineplot/base.hpp"
#include "jetstream/waterfall/base.hpp"

namespace Jetstream {

class Engine : public Graph {
public:
    Result compute();
    Result present();

private:
    std::mutex m;
    std::condition_variable access;
    std::atomic<bool> waiting{false};
};

template<typename T>
inline std::shared_ptr<T> Factory(Locale L, const typename T::Config & config) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<typename T::CPU>(config);
        case Jetstream::Locale::CUDA:
            return std::make_shared<typename T::CUDA>(config);
    }
}

} // namespace Jetstream

#endif
