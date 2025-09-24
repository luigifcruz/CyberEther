#include "jetstream/memory2/tensor.hh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "jetstream/logger.hh"

namespace Jetstream::mem2 {

namespace {
std::atomic<Index> g_tensor_counter{1};
}

struct Tensor::Impl {
    DataType dtype = DataType::None;
    Device current_device = Device::None;
    Device root_device = Device::None;
    std::unordered_map<Device, Buffer> buffers;
    Index identifier = 0;

    struct Layout {
        Shape shape;
        Shape stride;
        Shape shape_minus_one;
        Shape backstride;
        U64 offset = 0;
        U64 size = 0;
        U64 size_bytes = 0;
        U64 offset_bytes = 0;
        U64 element_size = 0;
        bool contiguous = true;

        void computeDefaultStrides();
        void updateCache();
        void initialize(const Shape& shape, U64 element_size);
        Result expand(Index axis);
        Result squeeze(Index axis);
        Result reshape(const Shape& new_shape);
        Result broadcast(const Shape& target_shape);
        Result slice(const std::vector<Token>& slice);
        U64 shapeToOffset(const std::initializer_list<U64>& coordinates) const;
    } layout;
};

void Tensor::Impl::Layout::computeDefaultStrides() {
    const auto rank = shape.size();
    stride.resize(rank);
    if (rank == 0) {
        return;
    }

    U64 stride_val = 1;
    for (std::size_t i = rank; i-- > 0;) {
        stride[i] = stride_val;
        stride_val *= shape[i];
    }
}

void Tensor::Impl::Layout::updateCache() {
    const auto rank = shape.size();
    shape_minus_one.resize(rank);
    backstride.resize(rank);

    if (rank == 0) {
        size = 0;
        size_bytes = 0;
        offset_bytes = offset * element_size;
        return;
    }

    size = 1;
    for (std::size_t i = 0; i < rank; ++i) {
        const U64 dim = shape[i];
        shape_minus_one[i] = dim > 0 ? dim - 1 : 0;
        const U64 stride_val = stride.size() > i ? stride[i] : 0;
        backstride[i] = stride_val * shape_minus_one[i];

        size *= dim;
    }

    size_bytes = size * element_size;
    offset_bytes = offset * element_size;
}

void Tensor::Impl::Layout::initialize(const Shape& shape, U64 element_size) {
    this->shape = shape;
    this->element_size = element_size;
    offset = 0;
    contiguous = true;
    computeDefaultStrides();
    updateCache();
}

Result Tensor::Impl::Layout::expand(Index axis) {
    if (axis > shape.size()) {
        JST_ERROR("[MEM2:TENSOR] expand_dims axis {} out of range {}.", axis, shape.size());
        return Result::ERROR;
    }

    U64 new_stride = 1;
    if (!stride.empty()) {
        if (axis == 0) {
            new_stride = stride.front();
        } else {
            new_stride = stride[axis - 1];
        }
    }

    shape.insert(shape.begin() + axis, 1);
    stride.insert(stride.begin() + axis, new_stride);
    shape_minus_one.insert(shape_minus_one.begin() + axis, 0);
    backstride.insert(backstride.begin() + axis, 0);

    updateCache();
    return Result::SUCCESS;
}

Result Tensor::Impl::Layout::squeeze(Index axis) {
    if (axis >= shape.size()) {
        JST_ERROR("[MEM2:TENSOR] squeeze_dims axis {} out of range {}.", axis, shape.size());
        return Result::ERROR;
    }

    if (shape[axis] != 1) {
        JST_ERROR("[MEM2:TENSOR] Cannot squeeze dimension {} (size {}).", axis, shape[axis]);
        return Result::ERROR;
    }

    shape.erase(shape.begin() + axis);
    stride.erase(stride.begin() + axis);
    shape_minus_one.erase(shape_minus_one.begin() + axis);
    backstride.erase(backstride.begin() + axis);

    updateCache();
    return Result::SUCCESS;
}

Result Tensor::Impl::Layout::reshape(const Shape& new_shape) {
    if (new_shape.empty()) {
        JST_ERROR("[MEM2:TENSOR] Cannot reshape to an empty shape.");
        return Result::ERROR;
    }

    if (!contiguous) {
        JST_ERROR("[MEM2:TENSOR] Cannot reshape non-contiguous tensor.");
        return Result::ERROR;
    }

    const U64 original_size = size;
    U64 new_size = 1;
    for (const auto dim : new_shape) {
        if (dim == 0) {
            JST_ERROR("[MEM2:TENSOR] Reshape dimension cannot be zero.");
            return Result::ERROR;
        }
        new_size *= dim;
    }

    if (original_size != new_size) {
        JST_ERROR("[MEM2:TENSOR] Reshape size mismatch {} -> {}.", original_size, new_size);
        return Result::ERROR;
    }

    shape = new_shape;
    computeDefaultStrides();
    contiguous = true;
    updateCache();
    return Result::SUCCESS;
}

Result Tensor::Impl::Layout::broadcast(const Shape& target_shape) {
    if (target_shape.size() < shape.size()) {
        JST_ERROR("[MEM2:TENSOR] Cannot broadcast shape {} -> {}.", shape, target_shape);
        return Result::ERROR;
    }

    const auto diff = target_shape.size() - shape.size();
    for (std::size_t i = 0; i < diff; ++i) {
        JST_CHECK(expand(0));
    }

    bool contiguous_flag = contiguous;
    Shape new_shape(target_shape.size());
    Shape new_stride(target_shape.size());

    for (std::size_t i = 0; i < target_shape.size(); ++i) {
        const U64 current_dim = shape[i];
        const U64 target_dim = target_shape[i];

        if (current_dim == target_dim) {
            new_shape[i] = current_dim;
            new_stride[i] = stride[i];
        } else if (current_dim == 1) {
            new_shape[i] = target_dim;
            new_stride[i] = 0;
            contiguous_flag = false;
        } else if (target_dim == 1) {
            new_shape[i] = current_dim;
            new_stride[i] = stride[i];
        } else {
            JST_ERROR("[MEM2:TENSOR] Cannot broadcast dimension {} -> {} (axis {}).", current_dim, target_dim, i);
            return Result::ERROR;
        }
    }

    shape = std::move(new_shape);
    stride = std::move(new_stride);
    contiguous = contiguous_flag;
    updateCache();
    return Result::SUCCESS;
}

Result Tensor::Impl::Layout::slice(const std::vector<Token>& slice) {
    Shape new_shape;
    Shape new_stride;
    U64 offset_val = offset;
    Index dim = 0;
    bool ellipsis_used = false;

    for (const auto& token : slice) {
        switch (token.get_type()) {
            case Token::Type::Number: {
                if (dim >= shape.size()) {
                    JST_ERROR("[MEM2:TENSOR] Slice index exceeds dimensions.");
                    return Result::ERROR;
                }

                const U64 index = token.get_a();
                if (index >= shape[dim]) {
                    JST_ERROR("[MEM2:TENSOR] Slice index {} out of range {}.", index, shape[dim]);
                    return Result::ERROR;
                }

                offset_val += index * stride[dim];
                dim++;
                break;
            }
            case Token::Type::Colon: {
                if (dim >= shape.size()) {
                    JST_ERROR("[MEM2:TENSOR] Slice index exceeds dimensions.");
                    return Result::ERROR;
                }

                const U64 start = token.get_a();
                U64 end = token.get_b();
                const U64 step = token.get_c();

                if (end == 0) {
                    end = shape[dim];
                }

                if (step == 0) {
                    JST_ERROR("[MEM2:TENSOR] Slice step cannot be zero.");
                    return Result::ERROR;
                }

                if (start >= shape[dim] || end > shape[dim]) {
                    JST_ERROR("[MEM2:TENSOR] Slice range [{}:{}] exceeds dimension {}.", start, end, shape[dim]);
                    return Result::ERROR;
                }

                if (start >= end) {
                    JST_ERROR("[MEM2:TENSOR] Slice start must be less than end.");
                    return Result::ERROR;
                }

                new_shape.push_back((end - start + step - 1) / step);
                new_stride.push_back(stride[dim] * step);
                offset_val += start * stride[dim];
                dim++;
                break;
            }
            case Token::Type::Ellipsis: {
                if (ellipsis_used) {
                    JST_ERROR("[MEM2:TENSOR] Ellipsis can only appear once in slice.");
                    return Result::ERROR;
                }
                ellipsis_used = true;

                const U64 remaining_dims = shape.size() - (slice.size() - 1);
                while (dim < remaining_dims) {
                    new_shape.push_back(shape[dim]);
                    new_stride.push_back(stride[dim]);
                    dim++;
                }
                break;
            }
        }
    }

    if (!ellipsis_used) {
        while (dim < shape.size()) {
            new_shape.push_back(shape[dim]);
            new_stride.push_back(stride[dim]);
            dim++;
        }
    }

    bool is_contiguous = true;
    if (!new_shape.empty()) {
        U64 expected_stride = 1;
        for (std::size_t i = new_shape.size(); i-- > 0;) {
            if (new_stride[i] != expected_stride) {
                is_contiguous = false;
                break;
            }
            expected_stride *= new_shape[i];
        }
    }

    shape = std::move(new_shape);
    stride = std::move(new_stride);
    offset = offset_val;
    contiguous = is_contiguous;
    updateCache();

    return Result::SUCCESS;
}

U64 Tensor::Impl::Layout::shapeToOffset(const std::initializer_list<U64>& coordinates) const {
    if (shape.empty()) {
        return 0;
    }

    if (coordinates.size() < shape.size()) {
        JST_ERROR("[MEM2:TENSOR] Coordinate rank {} smaller than tensor rank {}.", coordinates.size(), shape.size());
        return 0;
    }

    const std::size_t pad = coordinates.size() - shape.size();
    U64 index = offset;
    auto it = coordinates.begin();
    std::advance(it, pad);
    for (std::size_t i = 0; i < shape.size(); ++i, ++it) {
        index += (*it) * stride[i];
    }
    return index;
}

Tensor::Tensor() {
    ensure_impl();
}

Tensor::Tensor(const Device& device, const DataType& dtype, const Shape& shape) {
    ensure_impl();
    auto status = create(device, dtype, shape);
    if (status != Result::SUCCESS) {
        JST_ERROR("[MEM2:TENSOR] Failed to construct tensor {} on {}.", DataTypeName(dtype), device);
    }
}

Tensor::Tensor(const Device& device, const Tensor& source) {
    ensure_impl();
    auto status = create(device, source);
    if (status != Result::SUCCESS) {
        JST_ERROR("[MEM2:TENSOR] Failed to map tensor {} on {}.", DataTypeName(source.dtype()), device);
    }
}

Tensor::~Tensor() = default;

void Tensor::ensure_impl() {
    if (!impl) {
        impl = std::make_shared<Tensor::Impl>();
    }
}

Result Tensor::create(const Device& device, const DataType& dtype, const Shape& shape) {
    ensure_impl();

    if (dtype == DataType::None) {
        JST_ERROR("[MEM2:TENSOR] Invalid data type.");
        return Result::ERROR;
    }

    if (shape.empty()) {
        JST_ERROR("[MEM2:TENSOR] Shape cannot be empty.");
        return Result::ERROR;
    }

    impl->dtype = dtype;
    impl->layout.initialize(shape, DataTypeSize(dtype));
    impl->buffers.clear();
    impl->root_device = device;
    impl->current_device = device;
    impl->identifier = g_tensor_counter.fetch_add(1, std::memory_order_relaxed);

    Buffer buffer;
    JST_CHECK(buffer.create(device, impl->layout.size_bytes));
    impl->buffers.emplace(device, std::move(buffer));

    return Result::SUCCESS;
}

Result Tensor::create(const Device& device, const Tensor& source) {
    ensure_impl();

    Buffer buffer;
    JST_CHECK(buffer.create(device, source.buffer()));
    impl->buffers.clear();
    impl->buffers.emplace(device, std::move(buffer));

    impl->dtype = source.dtype();
    impl->layout.initialize(source.shape(), DataTypeSize(source.dtype()));
    impl->root_device = source.native_device();
    impl->current_device = device;
    impl->identifier = g_tensor_counter.fetch_add(1, std::memory_order_relaxed);

    return Result::SUCCESS;
}

const Device& Tensor::device() const {
    static const Device kNone = Device::None;
    return impl ? impl->current_device : kNone;
}

const Device& Tensor::native_device() const {
    static const Device kNone = Device::None;
    return impl ? impl->root_device : kNone;
}

const DataType& Tensor::dtype() const {
    static const DataType kNone = DataType::None;
    return impl ? impl->dtype : kNone;
}

const U64& Tensor::size() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.size : kZero;
}

const U64& Tensor::size_bytes() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.size_bytes : kZero;
}

const U64& Tensor::element_size() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.element_size : kZero;
}

bool Tensor::contiguous() const {
    return impl && impl->layout.contiguous;
}

bool Tensor::empty() const {
    return impl ? impl->layout.size == 0 : true;
}

bool Tensor::valid_shape() const {
    return impl ? !impl->layout.shape.empty() : false;
}

const U64& Tensor::offset() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.offset : kZero;
}

const U64& Tensor::offset_bytes() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.offset_bytes : kZero;
}

const Shape& Tensor::shape() const {
    static const Shape kEmpty{};
    return impl ? impl->layout.shape : kEmpty;
}

const Shape& Tensor::stride() const {
    static const Shape kEmpty{};
    return impl ? impl->layout.stride : kEmpty;
}

const Shape& Tensor::shape_minus_one() const {
    static const Shape kEmpty{};
    return impl ? impl->layout.shape_minus_one : kEmpty;
}

const Shape& Tensor::backstride() const {
    static const Shape kEmpty{};
    return impl ? impl->layout.backstride : kEmpty;
}

const U64& Tensor::shape(Index idx) const {
    return impl->layout.shape.at(idx);
}

const U64& Tensor::stride(Index idx) const {
    return impl->layout.stride.at(idx);
}

Index Tensor::rank() const {
    return impl ? static_cast<Index>(impl->layout.shape.size()) : 0;
}

Index Tensor::ndims() const {
    return impl ? static_cast<Index>(impl->layout.shape.size()) : 0;
}

U64 Tensor::shape_to_offset(const std::initializer_list<U64>& coordinates) const {
    if (!impl) {
        return 0;
    }
    return impl->layout.shapeToOffset(coordinates);
}

Result Tensor::expand_dims(Index axis) {
    if (!impl) {
        return Result::ERROR;
    }
    return impl->layout.expand(axis);
}

Result Tensor::squeeze_dims(Index axis) {
    if (!impl) {
        return Result::ERROR;
    }
    return impl->layout.squeeze(axis);
}

Result Tensor::reshape(const Shape& new_shape) {
    if (!impl) {
        return Result::ERROR;
    }
    return impl->layout.reshape(new_shape);
}

Result Tensor::broadcast_to(const Shape& new_shape) {
    if (!impl) {
        return Result::ERROR;
    }
    return impl->layout.broadcast(new_shape);
}

Result Tensor::slice(const std::vector<Token>& tokens) {
    if (!impl) {
        return Result::ERROR;
    }
    return impl->layout.slice(tokens);
}

Result Tensor::has_device(const Device& device) {
    ensure_impl();

    if (impl->buffers.contains(device)) {
        return Result::SUCCESS;
    }

    if (!impl->buffers.contains(impl->root_device)) {
        JST_ERROR("[MEM2:TENSOR] Root buffer is not initialized.");
        return Result::ERROR;
    }

    Buffer clone;
    JST_CHECK(clone.create(device, impl->buffers.at(impl->root_device)));
    impl->buffers.emplace(device, std::move(clone));

    return Result::SUCCESS;
}

Result Tensor::copy_from(const Tensor& source) {
    ensure_impl();

    if (!contiguous()) {
        JST_ERROR("[MEM2:TENSOR] Source tensor is not contiguous.");
        return Result::ERROR;
    }

    if (source.size() != size()) {
        JST_ERROR("[MEM2:BUFFER] Source tensor size does not match destination tensor size.");
        return Result::ERROR;
    }

    // Try to map the buffer to the source device.
    JST_CHECK(has_device(source.device()));

    // Copy the buffer from the source device to the current device.
    JST_CHECK(impl->buffers.at(source.device()).copy_from(source.buffer()));

    return Result::SUCCESS;
}

void* Tensor::data() {
    if (!impl || impl->current_device == Device::None) {
        return nullptr;
    }

    if (!impl->buffers.contains(impl->current_device)) {
        return nullptr;
    }

    auto& buf = impl->buffers.at(impl->current_device);
    void* base = buf.data();
    if (!base) {
        return nullptr;
    }

    if (impl->current_device == Device::CPU) {
        auto* bytes = static_cast<std::uint8_t*>(base);
        return bytes + impl->layout.offset_bytes;
    }

    return base;
}

const void* Tensor::data() const {
    if (!impl || impl->current_device == Device::None) {
        return nullptr;
    }

    if (!impl->buffers.contains(impl->current_device)) {
        return nullptr;
    }

    const auto& buf = impl->buffers.at(impl->current_device);
    const void* base = buf.data();
    if (!base) {
        return nullptr;
    }

    if (impl->current_device == Device::CPU) {
        auto* bytes = static_cast<const std::uint8_t*>(base);
        return bytes + impl->layout.offset_bytes;
    }

    return base;
}

const Buffer& Tensor::buffer() const {
    if (!impl || impl->current_device == Device::None) {
        throw std::runtime_error("Tensor buffer requested before initialization");
    }
    const auto it = impl->buffers.find(impl->current_device);
    if (it == impl->buffers.end()) {
        JST_ERROR("[MEM2:TENSOR] Buffer for device {} is not materialized.", impl->current_device);
        throw std::runtime_error("Tensor buffer not materialized for device");
    }
    return it->second;
}

Buffer& Tensor::buffer() {
    if (!impl || impl->current_device == Device::None) {
        throw std::runtime_error("Tensor buffer requested before initialization");
    }
    const auto it = impl->buffers.find(impl->current_device);
    if (it == impl->buffers.end()) {
        JST_ERROR("[MEM2:TENSOR] Buffer for device {} is not materialized.", impl->current_device);
        throw std::runtime_error("Tensor buffer not materialized for device");
    }
    return it->second;
}

const Index& Tensor::id() const {
    return impl->identifier;
}

}  // namespace Jetstream::mem2
