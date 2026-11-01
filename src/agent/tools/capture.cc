#include "support/base.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/render/base/texture.hh"
#include "tools/base64.hh"

#include <stb_image_write.h>

#include <algorithm>
#include <cmath>

namespace Jetstream::Agent::Detail {
namespace {

constexpr F64 kMaxLongSide = 2048.0;
constexpr F64 kMaxShortSide = 768.0;
constexpr int kJpegQuality = 90;

std::vector<uint8_t> Downscale(const std::vector<uint8_t>& source,
                               U64 width,
                               U64 height,
                               U64 targetWidth,
                               U64 targetHeight) {
    std::vector<uint8_t> target(targetWidth * targetHeight * 4);

    for (U64 y = 0; y < targetHeight; ++y) {
        const U64 y0 = y * height / targetHeight;
        const U64 y1 = std::max(y0 + 1, (y + 1) * height / targetHeight);

        for (U64 x = 0; x < targetWidth; ++x) {
            const U64 x0 = x * width / targetWidth;
            const U64 x1 = std::max(x0 + 1, (x + 1) * width / targetWidth);

            U64 sum[4] = {0, 0, 0, 0};
            for (U64 sy = y0; sy < y1; ++sy) {
                const uint8_t* row = source.data() + (sy * width + x0) * 4;
                for (U64 sx = x0; sx < x1; ++sx, row += 4) {
                    for (U64 c = 0; c < 4; ++c) {
                        sum[c] += row[c];
                    }
                }
            }

            const U64 count = (y1 - y0) * (x1 - x0);
            uint8_t* out = target.data() + (y * targetWidth + x) * 4;
            for (U64 c = 0; c < 4; ++c) {
                out[c] = static_cast<uint8_t>(sum[c] / count);
            }
        }
    }

    return target;
}

std::string EncodeJpeg(const std::vector<uint8_t>& pixels, U64 width, U64 height) {
    std::vector<uint8_t> jpeg;
    const auto writer = [](void* context, void* data, int length) {
        auto* out = static_cast<std::vector<uint8_t>*>(context);
        const auto* bytes = static_cast<const uint8_t*>(data);
        out->insert(out->end(), bytes, bytes + length);
    };

    const int written = stbi_write_jpg_to_func(writer,
                                               &jpeg,
                                               static_cast<int>(width),
                                               static_cast<int>(height),
                                               4,
                                               pixels.data(),
                                               kJpegQuality);
    if (!written) {
        return {};
    }

    return jst::fmt::format("data:image/jpeg;base64,{}", Base64::Encode(jpeg.data(), jpeg.size()));
}

Tools::Output CaptureView(const Invocation& call) {
    const auto& flowgraph = call.flowgraph;
    const auto& args = call.args;

    if (!flowgraph) {
        return Response::Failure("No flowgraph is currently selected.");
    }

    if (!flowgraph->view().canCapture()) {
        return Response::Failure("The current graphics backend cannot capture views.");
    }

    const std::string name = args.value("name", std::string{});
    if (name.empty() || !flowgraph->view().has(name)) {
        return Response::Failure(Lookup::Missing(flowgraph, name));
    }

    std::vector<std::shared_ptr<Module::Surface>> surfaces;
    if (flowgraph->view().surfaces(name, surfaces) != Result::SUCCESS) {
        return Response::Failure(jst::fmt::format("Could not access surfaces of block '{}'.", name));
    }

    const std::string surfaceId = args.value("surface", std::string{});
    std::shared_ptr<const Render::Texture> texture;
    std::string matchedId;
    std::vector<std::string> availableIds;

    for (const auto& surface : surfaces) {
        for (const auto& manifest : surface->manifests()) {
            availableIds.push_back(manifest.id);

            if (!texture && (surfaceId.empty() || manifest.id == surfaceId)) {
                texture = manifest.surface;
                matchedId = manifest.id;
            }
        }
    }

    if (!texture) {
        if (availableIds.empty()) {
            return Response::Failure(jst::fmt::format("Block '{}' has no rendered view to capture.", name));
        }
        return Response::Failure(jst::fmt::format("Block '{}' has no surface '{}'. Available surfaces: {}",
                                                  name, surfaceId, Response::Dump(availableIds)));
    }

    const auto& config = texture->getConfig();
    const bool rgba = config.pfmt == Render::Texture::PixelFormat::RGBA;
    const bool ui8 = config.ptype == Render::Texture::PixelType::UI8;
    if (!rgba || !ui8) {
        return Response::Failure(jst::fmt::format("The view of block '{}' uses an unsupported pixel format.", name));
    }

    Extent2D<U64> size{};
    std::vector<uint8_t> pixels;
    const auto captured = flowgraph->view().capture(texture, size, pixels);

    if (size.x == 0 || size.y == 0) {
        return Response::Failure(jst::fmt::format("The view of block '{}' has not been rendered yet.", name));
    }

    if (captured != Result::SUCCESS) {
        return Response::Failure(jst::fmt::format("Capturing the view of block '{}' failed.", name));
    }

    for (U64 i = 3; i < pixels.size(); i += 4) {
        pixels[i] = 255;
    }

    const F64 longSide = static_cast<F64>(std::max(size.x, size.y));
    const F64 shortSide = static_cast<F64>(std::min(size.x, size.y));
    const F64 scale = std::min({1.0, kMaxLongSide / longSide, kMaxShortSide / shortSide});

    const U64 width = std::max<U64>(1, static_cast<U64>(std::lround(static_cast<F64>(size.x) * scale)));
    const U64 height = std::max<U64>(1, static_cast<U64>(std::lround(static_cast<F64>(size.y) * scale)));

    if (width != size.x || height != size.y) {
        pixels = Downscale(pixels, size.x, size.y, width, height);
    }

    const auto image = EncodeJpeg(pixels, width, height);
    if (image.empty()) {
        return Response::Failure("Failed to encode the captured view as JPEG.");
    }

    const Json body = {
        {"status", "success"},
        {"name", name},
        {"surface", matchedId},
        {"width", size.x},
        {"height", size.y},
    };
    return Response::Reply(body, {image});
}

}  // namespace

std::vector<Entry> Toolbox::Capture() {
    return {{
        {
            "inspect_block_view",
            "Capture one rendered surface from a visual block and return its surface id and dimensions. The image itself is attached to the conversation as the next user message.",
            R"({"type":"object","required":["name"],"properties":{"name":{"type":"string","description":"Exact block name from inspect_flowgraph."},"surface":{"type":"string","description":"Optional surface id from inspect_block."}},"additionalProperties":false})",
            {.needsGraph = true, .needsCapture = true},
        },
        CaptureView,
    }};
}

}  // namespace Jetstream::Agent::Detail
