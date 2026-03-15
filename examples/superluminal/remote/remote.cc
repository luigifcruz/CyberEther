#include <jetstream/superluminal.hh>

#include <random>

using namespace Jetstream;

static Result App() {
    std::cout << "Welcome to Superluminal Remote!" << std::endl;

    Tensor data(DeviceType::CPU, TypeToDataType<F32>(), {1, 8192});
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<F32> dist(0.0f, 1.0f);

    for (U64 j = 0; j < data.shape(1); j++) {
        data.at<F32>(0, j) = dist(rng);
    }

    Superluminal::InstanceConfig config;
    config.remote = true;
    JST_CHECK(Superluminal::Initialize(config));
    JST_CHECK(Superluminal::PrintRemoteInfo());

    JST_CHECK(Superluminal::Plot("Remote Demo", {{1}}, {
        .buffer = data,
        .type = Superluminal::Type::Line,
        .source = Superluminal::Domain::Time,
        .display = Superluminal::Domain::Frequency,
        .options = {},
    }));

    JST_CHECK(Superluminal::Show());

    std::cout << "Goodbye from Superluminal Remote!" << std::endl;

    return Result::SUCCESS;
}

int main() {
    return App() == Result::SUCCESS ? 0 : 1;
}
