#include <jetstream/superluminal.hh>
#include <random>

using namespace Jetstream;

static Result App() {
    std::cout << "Welcome to Superluminal!" << std::endl;

    Tensor data(DeviceType::CPU, TypeToDataType<F32>(), {1, 8192});
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<F32> dist(0.0f, 1.0f);

    for (U64 j = 0; j < data.shape(1); j++) {
        data.at<F32>(0, j) = dist(rng);
    }

    JST_CHECK(Superluminal::Plot("Random", {{1}}, {
        .buffer = data,
        .type = Superluminal::Type::Line,
        .source = Superluminal::Domain::Time,
        .display = Superluminal::Domain::Frequency,
        .options = {},
    }));

    JST_CHECK(Superluminal::Show());

    std::cout << "Goodbye from Superluminal!" << std::endl;

    return Result::SUCCESS;
}

int main() {
    return App() == Result::SUCCESS ? 0 : 1;
}
