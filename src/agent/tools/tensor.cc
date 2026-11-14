#include "support/base.hh"
#include "jetstream/tools/numeric.hh"
#include "tools/text.hh"
#include "tools/vector_types.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <optional>
#include <type_traits>

namespace Jetstream::Agent::Detail {
namespace {

constexpr U64 kMaxValues = 256;
constexpr U64 kMaxStatsElements = U64{1} << 20;
constexpr U64 kMaxSliceTokens = 32;
constexpr U64 kMaxAttributes = 32;
constexpr U64 kMaxAttributeBytes = 16384;
constexpr U64 kMaxAttributeDepth = 4;
constexpr U64 kMaxAttributeEntries = 32;
constexpr U64 kMaxAttributeKeyBytes = 64;
constexpr U64 kMaxAttributeStringBytes = 256;
constexpr U64 kMaxAttributeSlotBytes = (kMaxAttributeKeyBytes * 6) + 4 + 51;
static_assert(2 + (kMaxAttributes * kMaxAttributeSlotBytes) <= kMaxAttributeBytes);

struct Request {
    std::string name;
    std::string port;
    bool output = true;
    std::vector<Token> slice;
    U64 values = 0;
    bool stats = true;
};

template<typename T>
struct Scalar {
    using Part = T;
    static constexpr bool Complex = false;
};

template<typename T>
struct Scalar<std::complex<T>> {
    using Part = T;
    static constexpr bool Complex = true;
};

template<typename T>
Json Number(const T value) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(value)) {
            return "NaN";
        }
        if (std::isinf(value)) {
            return value > 0 ? "Infinity" : "-Infinity";
        }
        return static_cast<F64>(value);
    } else if constexpr (std::is_signed_v<T>) {
        return static_cast<I64>(value);
    } else {
        return static_cast<U64>(value);
    }
}

void Accumulate(F64& mean, const F64 value, const U64 count) {
    const auto samples = static_cast<F64>(count);
    mean = mean * ((samples - 1.0) / samples) + value / samples;
}

template<typename Part>
struct Component {
    Part minimum = {};
    Part maximum = {};
    F64 mean = 0.0;

    void add(const Part value, const U64 count) {
        minimum = count == 1 ? value : std::min(minimum, value);
        maximum = count == 1 ? value : std::max(maximum, value);
        Accumulate(mean, static_cast<F64>(value), count);
    }

    Json json(const U64 count) const {
        Json body = Json::object();
        body["min"] = count > 0 ? Number(minimum) : Json(nullptr);
        body["max"] = count > 0 ? Number(maximum) : Json(nullptr);
        body["mean"] = count > 0 ? Number(mean) : Json(nullptr);
        return body;
    }
};

class Elements {
 public:
    explicit Elements(const Tensor& view) : shape(view.shape()), stride(view.stride()) {}

    U64 index(U64 position) const {
        U64 result = 0;
        for (auto axis = shape.size(); axis-- > 0;) {
            result += (position % shape[axis]) * stride[axis];
            position /= shape[axis];
        }
        return result;
    }

 private:
    const Shape& shape;
    const Shape& stride;
};

U64 SampleStep(const Tensor& view) {
    if (view.size() <= kMaxStatsElements) {
        return 1;
    }

    U64 step = view.size() / kMaxStatsElements + (view.size() % kMaxStatsElements != 0 ? 1 : 0);
    const auto coprime = [&step](const U64 extent) {
        return std::gcd(step, extent) == 1;
    };
    while (!std::all_of(view.shape().begin(), view.shape().end(), coprime)) {
        step += 1;
    }
    return step;
}

template<typename T>
Json Statistics(const T* data, const Tensor& view) {
    using Part = typename Scalar<T>::Part;
    constexpr bool complex = Scalar<T>::Complex;

    const Elements elements(view);
    const U64 selected = view.size();
    const U64 step = SampleStep(view);
    const U64 examined = selected == 0 ? 0 : (selected - 1) / step + 1;

    Component<Part> real;
    Component<Part> imag;
    F64 minimumMagnitude = 0.0;
    F64 maximumMagnitude = 0.0;
    F64 power = 0.0;
    U64 finite = 0;
    U64 nan = 0;
    U64 infinite = 0;

    for (U64 sample = 0; sample < examined; ++sample) {
        const T& value = data[elements.index(sample * step)];

        Part re;
        Part im = {};
        if constexpr (complex) {
            re = value.real();
            im = value.imag();
        } else {
            re = value;
        }

        if constexpr (std::is_floating_point_v<Part>) {
            if (std::isnan(re) || std::isnan(im)) {
                nan += 1;
                continue;
            }
            if (std::isinf(re) || std::isinf(im)) {
                infinite += 1;
                continue;
            }
        }

        finite += 1;
        real.add(re, finite);

        const auto reWide = static_cast<F64>(re);
        const auto imWide = static_cast<F64>(im);
        if constexpr (complex) {
            imag.add(im, finite);
            const F64 magnitude = std::hypot(reWide, imWide);
            minimumMagnitude = finite == 1 ? magnitude : std::min(minimumMagnitude, magnitude);
            maximumMagnitude = finite == 1 ? magnitude : std::max(maximumMagnitude, magnitude);
        }
        Accumulate(power, reWide * reWide + imWide * imWide, finite);
    }

    Json stats = Json::object();
    stats["exact"] = examined == selected;
    stats["selected_elements"] = selected;
    stats["examined_elements"] = examined;
    stats["sample_step"] = step;
    stats["finite_elements"] = finite;
    stats["nan_elements"] = nan;
    stats["infinite_elements"] = infinite;
    stats["real"] = real.json(finite);
    if constexpr (complex) {
        stats["imag"] = imag.json(finite);
        stats["magnitude"] = Json::object();
        stats["magnitude"]["min"] = finite > 0 ? Number(minimumMagnitude) : Json(nullptr);
        stats["magnitude"]["max"] = finite > 0 ? Number(maximumMagnitude) : Json(nullptr);
    }
    stats["power"] = finite > 0 ? Number(power) : Json(nullptr);
    stats["rms"] = finite > 0 ? Number(std::sqrt(power)) : Json(nullptr);
    stats["power_db"] = finite > 0 ? Number(10.0 * std::log10(power)) : Json(nullptr);
    return stats;
}

template<typename T>
Json Values(const T* data, const Elements& elements, const U64 count) {
    Json values = Json::array();
    for (U64 position = 0; position < count; ++position) {
        const T& value = data[elements.index(position)];
        if constexpr (Scalar<T>::Complex) {
            values.push_back(Json::array({Number(value.real()), Number(value.imag())}));
        } else {
            values.push_back(Number(value));
        }
    }
    return values;
}

template<typename T>
void ReadElements(const Tensor& view, const Request& request, Json& result) {
    const auto* data = view.data<T>();

    if (request.values > 0) {
        result["values"] = Values(data, Elements(view), std::min(request.values, view.size()));
        result["values_truncated"] = request.values < view.size();
    }
    if (request.stats) {
        result["stats"] = Statistics(data, view);
    }
}

bool DispatchElements(const Tensor& view, const Request& request, Json& result) {
    switch (view.dtype()) {
        case DataType::F32: ReadElements<F32>(view, request, result); return true;
        case DataType::F64: ReadElements<F64>(view, request, result); return true;
        case DataType::I8: ReadElements<I8>(view, request, result); return true;
        case DataType::I16: ReadElements<I16>(view, request, result); return true;
        case DataType::I32: ReadElements<I32>(view, request, result); return true;
        case DataType::I64: ReadElements<I64>(view, request, result); return true;
        case DataType::U8: ReadElements<U8>(view, request, result); return true;
        case DataType::U16: ReadElements<U16>(view, request, result); return true;
        case DataType::U32: ReadElements<U32>(view, request, result); return true;
        case DataType::U64: ReadElements<U64>(view, request, result); return true;
        case DataType::CF32: ReadElements<CF32>(view, request, result); return true;
        case DataType::CF64: ReadElements<CF64>(view, request, result); return true;
        case DataType::CI8: ReadElements<CI8>(view, request, result); return true;
        case DataType::CI16: ReadElements<CI16>(view, request, result); return true;
        case DataType::CI32: ReadElements<CI32>(view, request, result); return true;
        case DataType::CI64: ReadElements<CI64>(view, request, result); return true;
        case DataType::CU8: ReadElements<CU8>(view, request, result); return true;
        case DataType::CU16: ReadElements<CU16>(view, request, result); return true;
        case DataType::CU32: ReadElements<CU32>(view, request, result); return true;
        case DataType::CU64: ReadElements<CU64>(view, request, result); return true;
        default: return false;
    }
}

class AttributeEncoder {
 public:
    Json encode(const Tensor& tensor, const std::vector<std::string>& keys) {
        const auto visit = [&](const std::string& key, const auto& callback) {
            JST_CHECK_THROW(tensor.visitAttribute(key, callback));
        };

        remaining = kMaxAttributeBytes - (keys.empty() ? 2 : keys.size() + 1);
        for (const auto& key : keys) {
            remaining -= Response::Dump(key).size() + 1 + BudgetBytes();
        }

        Json attributes = Json::object();
        std::vector<std::string> pending;
        std::vector<std::string> bulk;
        for (const auto& key : keys) {
            visit(key, [&](const std::any& value) {
                if (IsScalar(value)) {
                    place(attributes, key, value, 1);
                } else if (IsText(value)) {
                    pending.push_back(key);
                } else {
                    bulk.push_back(key);
                }
            });
        }
        pending.insert(pending.end(), bulk.begin(), bulk.end());

        for (U64 index = 0; index < pending.size(); ++index) {
            visit(pending[index], [&](const std::any& value) {
                place(attributes, pending[index], value, pending.size() - index);
            });
        }
        return attributes;
    }

 private:
    using Encoded = std::optional<Json>;

    static Json Omitted(const std::string& reason, const std::string& type) {
        return {{"omitted", reason}, {"type", type}};
    }

    static Json Truncated(const std::string& type, const U64 size, Json preview) {
        return {{"truncated", true}, {"type", type}, {"size", size}, {"preview", std::move(preview)}};
    }

    static U64 BudgetBytes() {
        return Response::Dump(Omitted("budget", "value")).size();
    }

    template<typename... Types>
    static bool HoldsAny(const std::any& value) {
        return ((value.type() == typeid(Types)) || ...);
    }

    static bool IsScalar(const std::any& value) {
        return !value.has_value() ||
               HoldsAny<std::nullptr_t, bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, CF32, CF64>(value);
    }

    static bool IsText(const std::any& value) {
        return HoldsAny<std::string, const char*>(value);
    }

    template<typename T>
    static Json Primitive(const T& value) {
        if constexpr (std::is_same_v<T, bool>) {
            return value;
        } else if constexpr (Scalar<T>::Complex) {
            return Json::array({Number(value.real()), Number(value.imag())});
        } else {
            return Number(value);
        }
    }

    void place(Json& attributes, const std::string& key, const std::any& value, const U64 pending) {
        const U64 placeholder = BudgetBytes();
        const U64 held = remaining - remaining / pending;
        remaining = remaining - held + placeholder;
        auto encoded = encodeValue(value, 0);
        if (encoded) {
            attributes[key] = std::move(*encoded);
        } else {
            remaining -= placeholder;
            attributes[key] = Omitted("budget", "value");
        }
        remaining += held;
    }

    bool reserve(const U64 bytes) {
        if (bytes > remaining) {
            return false;
        }
        remaining -= bytes;
        return true;
    }

    Encoded leaf(Json value) {
        if (!reserve(Response::Dump(value).size())) {
            return std::nullopt;
        }
        return value;
    }

    Encoded encodeValue(const std::any& value, const U64 depth) {
        if (!value.has_value() || value.type() == typeid(std::nullptr_t)) {
            return leaf(nullptr);
        }
        if (const auto* map = std::any_cast<Parser::Map>(&value)) {
            return object(*map, depth);
        }
        if (const auto* sequence = std::any_cast<Parser::Sequence>(&value)) {
            return array(*sequence, depth);
        }
        if (const auto* text = std::any_cast<std::string>(&value)) {
            return string(*text);
        }
        if (const auto* text = std::any_cast<const char*>(&value)) {
            return string(*text ? std::string_view(*text) : std::string_view());
        }

        Encoded encoded;
        if (scalar<bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, CF32, CF64>(value, encoded)) {
            return encoded;
        }

        if (VectorTypes::Visit(value, [&](const auto& entries) {
            encoded = array(entries, depth);
        })) {
            return encoded;
        }

        std::string text;
        return Parser::TryTypedToString(value, text) ? string(text) : leaf(Omitted("unsupported", "value"));
    }

    Encoded string(const std::string_view value) {
        if (value.size() <= kMaxAttributeStringBytes) {
            auto whole = leaf(std::string(value));
            if (whole) {
                return whole;
            }
        }
        if (!reserve(Response::Dump(Truncated("string", value.size(), "")).size())) {
            return std::nullopt;
        }

        U64 limit = std::min<U64>(value.size(), kMaxAttributeStringBytes);
        std::string preview(Text::Utf8Prefix(value, limit));
        U64 bytes = Response::Dump(preview).size() - 2;
        while (bytes > remaining) {
            limit = limit * remaining / bytes;
            preview = std::string(Text::Utf8Prefix(value, limit));
            bytes = Response::Dump(preview).size() - 2;
        }
        remaining -= bytes;
        return Truncated("string", value.size(), std::move(preview));
    }

    template<typename T>
    Encoded element(const T& value, const U64 depth) {
        if constexpr (std::is_same_v<T, std::any>) {
            return encodeValue(value, depth);
        } else if constexpr (std::is_same_v<T, Parser::Map>) {
            return object(value, depth);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return string(value);
        } else {
            return leaf(Primitive<T>(value));
        }
    }

    template<typename... Types>
    bool scalar(const std::any& value, Encoded& encoded) {
        const auto visit = [&]<typename T>() {
            const auto* entry = std::any_cast<T>(&value);
            if (entry) {
                encoded = leaf(Primitive<T>(*entry));
            }
            return entry != nullptr;
        };
        return (visit.template operator()<Types>() || ...);
    }

    template<typename T>
    Encoded array(const std::vector<T>& entries, const U64 depth) {
        if (entries.empty()) {
            return leaf(Json::array());
        }
        if (depth >= kMaxAttributeDepth) {
            return leaf({{"omitted", "depth"}, {"type", "array"}, {"size", entries.size()}});
        }

        const U64 envelope = Response::Dump(Truncated("array", entries.size(), Json::array())).size();
        if (!reserve(envelope)) {
            return std::nullopt;
        }

        Json items = Json::array();
        const U64 limit = std::min<U64>(entries.size(), kMaxAttributeEntries);
        for (U64 index = 0; index < limit; ++index) {
            const U64 separator = index > 0 ? 1 : 0;
            if (!reserve(separator)) {
                break;
            }
            auto entry = element<T>(entries[index], depth + 1);
            if (!entry) {
                remaining += separator;
                break;
            }
            items.push_back(std::move(*entry));
        }

        if (items.size() < entries.size()) {
            return Truncated("array", entries.size(), std::move(items));
        }
        remaining += envelope - 2;
        return items;
    }

    Encoded object(const Parser::Map& entries, const U64 depth) {
        if (entries.empty()) {
            return leaf(Json::object());
        }
        if (depth >= kMaxAttributeDepth) {
            return leaf({{"omitted", "depth"}, {"type", "object"}, {"size", entries.size()}});
        }

        const U64 envelope = Response::Dump(Truncated("object", entries.size(), Json::object())).size();
        if (!reserve(envelope)) {
            return std::nullopt;
        }

        Json items = Json::object();
        U64 visited = 0;
        bool complete = true;
        for (const auto& entry : entries) {
            if (visited++ == kMaxAttributeEntries) {
                complete = false;
                break;
            }
            if (entry.key.size() > kMaxAttributeKeyBytes) {
                complete = false;
                continue;
            }
            const U64 prefix = Response::Dump(entry.key).size() + 1 + (items.empty() ? 0 : 1);
            if (!reserve(prefix)) {
                complete = false;
                break;
            }
            auto encoded = encodeValue(entry.value, depth + 1);
            if (!encoded) {
                remaining += prefix;
                complete = false;
                break;
            }
            items[entry.key] = std::move(*encoded);
        }

        if (!complete) {
            return Truncated("object", entries.size(), std::move(items));
        }
        remaining += envelope - 2;
        return items;
    }

    U64 remaining = 0;
};

void AttributesJson(const Tensor& tensor, Json& body) {
    std::vector<std::string> keys;
    U64 omitted = 0;
    for (auto& key : tensor.attributeKeys()) {
        if (keys.size() == kMaxAttributes || key.size() > kMaxAttributeKeyBytes) {
            omitted += 1;
            continue;
        }
        keys.push_back(std::move(key));
    }

    body["attributes"] = AttributeEncoder().encode(tensor, keys);
    if (omitted > 0) {
        body["attributes_omitted"] = omitted;
    }
}

Json TensorJson(const Tensor& tensor) {
    Json body = Json::object();
    body["dtype"] = std::string(DataTypeToName(tensor.dtype()));
    body["device"] = GetDeviceName(tensor.device());
    body["native_device"] = GetDeviceName(tensor.nativeDevice());
    body["shape"] = tensor.shape();
    body["strides"] = tensor.stride();
    body["offset"] = tensor.offset();
    body["elements"] = tensor.size();
    body["element_size"] = tensor.elementSize();
    body["size_bytes"] = tensor.sizeBytes();
    body["contiguous"] = tensor.contiguous();
    AttributesJson(tensor, body);
    return body;
}

std::string CheckSlice(const std::vector<Token>& tokens, const Tensor& tensor) {
    const auto consuming = static_cast<U64>(std::count_if(tokens.begin(), tokens.end(), [](const Token& token) {
        return token.getType() != Token::Type::Ellipsis;
    }));
    const U64 rank = tensor.rank();
    if (consuming > rank) {
        return jst::fmt::format("Slice addresses {} axes but the tensor has rank {}.", consuming, rank);
    }

    U64 axis = 0;
    for (const auto& token : tokens) {
        if (token.getType() == Token::Type::Ellipsis) {
            axis += rank - consuming;
            continue;
        }

        const U64 extent = tensor.shape(axis);
        if (token.getType() == Token::Type::Number) {
            if (token.getA() >= extent) {
                return jst::fmt::format("Slice index {} is out of range for axis {} of extent {}.",
                                        token.getA(), axis, extent);
            }
        } else {
            const U64 stop = token.hasEnd() ? token.getB() : extent;
            if (token.getA() > extent || stop > extent) {
                return jst::fmt::format("Slice range {}:{} exceeds axis {} of extent {}.",
                                        token.getA(), stop, axis, extent);
            }
        }
        axis += 1;
    }

    return {};
}

bool InsideStorage(const Tensor& view) {
    if (view.size() == 0) {
        return true;
    }

    U64 last = view.offset();
    for (const auto extent : view.backstride()) {
        if (!detail::CheckedAdd(last, extent, last)) {
            return false;
        }
    }

    U64 bytes = 0;
    return detail::CheckedAdd(last, 1, last) &&
           detail::CheckedMultiply(last, view.elementSize(), bytes) &&
           bytes <= view.buffer().sizeBytes();
}

std::string MissingTensor(const TensorLink& link, const Request& request) {
    if (request.output) {
        return jst::fmt::format("Output '{}' of block '{}' has no tensor data.", request.port, request.name);
    }
    if (link.external.has_value()) {
        return jst::fmt::format("Input '{}' of block '{}' is connected but has no tensor data.",
                                request.port, request.name);
    }
    return jst::fmt::format("Input '{}' of block '{}' is not connected.", request.port, request.name);
}

void ReadLink(const TensorLink& link, const Request& request, Json& result) {
    if (!request.output && link.external.has_value()) {
        result["source"] = Serialize::Port(link.external->block, link.external->port);
    }

    if (!link.resolved()) {
        result["error"] = MissingTensor(link, request);
        return;
    }

    const auto& tensor = link.tensor;
    result["tensor"] = TensorJson(tensor);

    const auto invalid = CheckSlice(request.slice, tensor);
    if (!invalid.empty()) {
        result["error"] = invalid;
        return;
    }

    auto view = tensor.clone();
    if (!request.slice.empty() && view.slice(request.slice) != Result::SUCCESS) {
        result["error"] = "Slice exceeds the supported layout range.";
        return;
    }

    Json selection = Json::object();
    selection["shape"] = view.shape();
    selection["strides"] = view.stride();
    selection["offset"] = view.offset();
    selection["elements"] = view.size();
    result["selection"] = std::move(selection);

    if (request.values == 0 && !request.stats) {
        return;
    }

    if (tensor.device() != DeviceType::CPU || tensor.nativeDevice() != DeviceType::CPU) {
        result["error"] = jst::fmt::format("Tensor data on device '{}' cannot be read here. Metadata is included.",
                                           GetDeviceName(tensor.device()));
        result["code"] = "UNSUPPORTED_DEVICE";
        return;
    }

    if (!InsideStorage(view) || (view.size() > 0 && !view.data())) {
        result["error"] = "Tensor layout does not fit its storage. No data was read.";
        result["code"] = "INVALID_LAYOUT";
        return;
    }

    if (!DispatchElements(view, request, result)) {
        result["error"] = jst::fmt::format("Tensor dtype '{}' is not supported.", DataTypeToName(tensor.dtype()));
        result["code"] = "UNSUPPORTED_DTYPE";
    }
}

std::string ParseSliceBound(const Json& range, const char* key, U64& value) {
    if (!range.contains(key)) {
        return {};
    }
    if (!range[key].is_number_unsigned()) {
        return jst::fmt::format("Slice field '{}' must be a nonnegative integer.", key);
    }
    value = range[key].get<U64>();
    return {};
}

std::string ParseSlice(const Json& slice, std::vector<Token>& tokens) {
    if (!slice.is_array()) {
        return "'slice' must be an array with one entry per axis.";
    }
    if (slice.size() > kMaxSliceTokens) {
        return jst::fmt::format("'slice' accepts at most {} entries.", kMaxSliceTokens);
    }

    bool ellipsis = false;
    for (const auto& entry : slice) {
        if (entry.is_number_unsigned()) {
            tokens.emplace_back(entry.get<U64>());
            continue;
        }

        if (entry.is_string() && entry.get<std::string>() == "...") {
            if (ellipsis) {
                return "'slice' accepts one ellipsis.";
            }
            ellipsis = true;
            tokens.emplace_back("...");
            continue;
        }

        if (!entry.is_object()) {
            return "Slice entries must be a nonnegative integer, a range object, or \"...\".";
        }

        for (const auto& [key, _] : entry.items()) {
            if (key != "start" && key != "stop" && key != "step") {
                return jst::fmt::format("Unknown slice field '{}'.", key);
            }
        }

        U64 start = 0;
        U64 stop = 0;
        U64 step = 1;
        for (const auto& error : {ParseSliceBound(entry, "start", start),
                                  ParseSliceBound(entry, "stop", stop),
                                  ParseSliceBound(entry, "step", step)}) {
            if (!error.empty()) {
                return error;
            }
        }
        if (step == 0) {
            return "Slice field 'step' must be at least 1.";
        }
        tokens.emplace_back(start, stop, step, entry.contains("stop"));
    }

    return {};
}

std::string ParseRequest(const Json& args, Request& request) {
    for (const auto& [key, _] : args.items()) {
        if (key != "name" && key != "port" && key != "direction" && key != "slice" &&
            key != "values" && key != "stats") {
            return jst::fmt::format("Unknown argument '{}'.", key);
        }
    }

    if (!args.contains("name") || !args["name"].is_string()) {
        return "'name' must be a block name string.";
    }
    if (!args.contains("port") || !args["port"].is_string()) {
        return "'port' must be a port name string.";
    }
    request.name = args["name"].get<std::string>();
    request.port = args["port"].get<std::string>();

    if (args.contains("direction")) {
        const auto& direction = args["direction"];
        if (!direction.is_string() || (direction != "input" && direction != "output")) {
            return "'direction' must be 'input' or 'output'.";
        }
        request.output = direction == "output";
    }

    if (args.contains("values")) {
        const auto& values = args["values"];
        if (!values.is_number_unsigned() || values.get<U64>() > kMaxValues) {
            return jst::fmt::format("'values' must be an integer between 0 and {}.", kMaxValues);
        }
        request.values = values.get<U64>();
    }

    if (args.contains("stats")) {
        if (!args["stats"].is_boolean()) {
            return "'stats' must be a boolean.";
        }
        request.stats = args["stats"].get<bool>();
    }

    return args.contains("slice") ? ParseSlice(args["slice"], request.slice) : std::string{};
}

Json InspectTensor(const Graph& flowgraph, const Json& args) {
    if (!flowgraph) {
        return Response::Error("No flowgraph is currently selected.");
    }

    Request request;
    const auto invalid = ParseRequest(args, request);
    if (!invalid.empty()) {
        return Response::Error(invalid);
    }

    if (!flowgraph->view().has(request.name)) {
        return Response::Error(Lookup::Missing(flowgraph, request.name));
    }

    std::vector<Flowgraph::View::InterfaceEntry> ports;
    const auto listed = request.output ? flowgraph->view().interfaceOutputs(request.name, ports)
                                       : flowgraph->view().interfaceInputs(request.name, ports);
    if (listed != Result::SUCCESS) {
        return Response::Error(Lookup::Missing(flowgraph, request.name));
    }

    const std::string direction = request.output ? "output" : "input";
    if (!Serialize::HasPort(ports, request.port)) {
        Json body = Response::Error(jst::fmt::format("Block '{}' has no {} port '{}'.",
                                                     request.name, direction, request.port));
        body["available_ports"] = Serialize::PortNames(ports);
        return body;
    }

    Json result = Json::object();
    result["name"] = request.name;
    result["port"] = request.port;
    result["direction"] = direction;

    const auto read = flowgraph->view().readTensor(request.name, request.port, request.output,
                                                   [&](const TensorLink& link) {
        ReadLink(link, request, result);
        return Result::SUCCESS;
    });
    if (read != Result::SUCCESS) {
        if (read == Result::TIMEOUT) {
            result["error"] = "The flowgraph is busy computing or changing, so no tensor data was read. "
                              "Nothing changed. Retry shortly.";
            result["code"] = "TENSOR_BUSY";
            result["retriable"] = true;
            return result;
        }
        if (!flowgraph->view().has(request.name)) {
            return Response::Error(Lookup::Missing(flowgraph, request.name));
        }
        return Response::Error(jst::fmt::format("Could not read {} '{}' of block '{}'.",
                                                direction, request.port, request.name));
    }

    Flowgraph::View::BlockInfo info;
    if (!result.contains("tensor") && flowgraph->view().info(request.name, info) == Result::SUCCESS) {
        result["state"] = GetBlockStateName(info.state);
    }

    return result;
}

}  // namespace

std::vector<Entry> Toolbox::Tensors() {
    return {
        Entry::FromJson({
            "inspect_tensor",
            "Read a bounded, consistent snapshot of one input or output tensor of a block without "
            "changing the graph. Compute and rendering wait only while the read runs. Returns tensor "
            "metadata and attributes, the selection layout, statistics by default, and up to 256 "
            "values on request. Values come from the last compute, so a tensor can be all zeros "
            "before its first frame. The slice array selects per axis in row-major order. An integer "
            "fixes an axis, an object with start, stop, and step keeps a range, and \"...\" expands "
            "the remaining axes. Complex values are [real, imag]. Non-finite numbers are the strings "
            "NaN, Infinity, and -Infinity. Statistics skip non-finite elements. real and imag report "
            "min, max, and mean per part, and magnitude reports min and max for complex tensors. "
            "power is the mean squared magnitude, rms is its root, and power_db is its level in "
            "decibels relative to one. Selections above 1048576 elements are sampled at evenly spaced "
            "row-major positions. exact is then false and sample_step gives the spacing, so "
            "statistics describe only the examined elements. Narrow the slice for exact results. "
            "Only CPU storage is read. Other devices return metadata with an UNSUPPORTED_DEVICE error "
            "unless stats is false and values is 0. A read waits at most 500 milliseconds for compute "
            "to reach a safe point. A TENSOR_BUSY error with retriable true means no data was read and "
            "nothing changed, so the same call can be retried. Attributes are limited to 32 keys of up "
            "to 64 bytes, 4 nesting levels, 32 entries per container, and 256 bytes per string. The "
            "complete serialized attributes object never exceeds 16 KiB. Numbers, booleans, and nulls "
            "are always kept, and strings and containers share the remaining space. Larger values "
            "become objects with truncated, type, size, and preview, where the preview can be shorter "
            "than the limits when space runs out. Values that are too deep, over budget, or "
            "unsupported become objects with omitted and type. attributes_omitted counts attributes "
            "left out entirely.",
            R"({"type":"object","required":["name","port"],"properties":{"name":{"type":"string","description":"Exact block name from inspect_flowgraph."},"port":{"type":"string","description":"Exact port name from inspect_block."},"direction":{"type":"string","enum":["output","input"],"description":"Port direction, default output."},"slice":{"type":"array","maxItems":32,"description":"Per-axis selection. Unlisted trailing axes are kept whole.","items":{"oneOf":[{"type":"integer","minimum":0},{"type":"object","properties":{"start":{"type":"integer","minimum":0},"stop":{"type":"integer","minimum":0},"step":{"type":"integer","minimum":1}},"additionalProperties":false},{"type":"string","enum":["..."]}]}},"values":{"type":"integer","minimum":0,"maximum":256,"description":"Leading selected elements to return in row-major order, default 0."},"stats":{"type":"boolean","description":"Compute statistics over the selection, default true."}},"additionalProperties":false})",
            {.needsGraph = true},
        }, InspectTensor),
    };
}

}  // namespace Jetstream::Agent::Detail
