#include <jetstream/superluminal.hh>
#include <random>

using namespace Jetstream;

static Result App() {
    Tensor data(DeviceType::CPU, TypeToDataType<CF32>(), {1, 2048});

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> symbol(0, 3);
    std::normal_distribution<F32> noise(0.0f, 0.05f);

    // Unit-energy QPSK constellation points.
    const F32 amplitude = 0.70710678f;
    const CF32 points[4] = {
        {+amplitude, +amplitude},
        {+amplitude, -amplitude},
        {-amplitude, +amplitude},
        {-amplitude, -amplitude},
    };

    for (U64 j = 0; j < data.shape(1); j++) {
        const auto& point = points[symbol(rng)];
        data.at<CF32>(0, j) = {point.real() + noise(rng), point.imag() + noise(rng)};
    }

    JST_CHECK(Superluminal::Plot("QPSK", {{1}}, {
        .buffer = data,
        .type = Superluminal::Type::Scatter,
        .source = Superluminal::Domain::Time,
        .display = Superluminal::Domain::Time,
        .options = {},
    }));

    JST_CHECK(Superluminal::Show());

    return Result::SUCCESS;
}

int main() {
    return App() == Result::SUCCESS ? 0 : 1;
}
