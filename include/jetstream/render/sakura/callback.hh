#ifndef JETSTREAM_RENDER_SAKURA_CALLBACK_HH
#define JETSTREAM_RENDER_SAKURA_CALLBACK_HH

#include <functional>
#include <utility>

namespace Jetstream::Sakura {

template<typename Signature>
struct Callback : public std::function<Signature> {
    using std::function<Signature>::function;

    Callback() = default;
    Callback(const std::function<Signature>& function) : std::function<Signature>(function) {}
    Callback(std::function<Signature>&& function) : std::function<Signature>(std::move(function)) {}

    bool operator==(const Callback& other) const {
        return static_cast<bool>(*this) == static_cast<bool>(other);
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CALLBACK_HH
