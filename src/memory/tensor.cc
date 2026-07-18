#include "jetstream/memory/tensor.hh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"

namespace Jetstream {

namespace {
std::atomic<Index> g_tensor_counter{1};

bool PointerRangesOverlap(const void* lhs, const U64 lhsSize,
                          const void* rhs, const U64 rhsSize) {
    if (!lhs || !rhs || lhsSize == 0 || rhsSize == 0) {
        return false;
    }

    const auto lhsStart = reinterpret_cast<std::uintptr_t>(lhs);
    const auto rhsStart = reinterpret_cast<std::uintptr_t>(rhs);
    const auto limit = std::numeric_limits<std::uintptr_t>::max();
    if (lhsSize > limit - lhsStart || rhsSize > limit - rhsStart) {
        return true;
    }

    const auto lhsEnd = lhsStart + static_cast<std::uintptr_t>(lhsSize);
    const auto rhsEnd = rhsStart + static_cast<std::uintptr_t>(rhsSize);
    return lhsStart < rhsEnd && rhsStart < lhsEnd;
}
}

struct Tensor::Impl {
    struct Storage {
        DeviceType rootDevice = DeviceType::None;
        std::shared_ptr<Storage> backingStorage;
        std::vector<std::shared_ptr<void>> backingOwners;
        std::unordered_map<DeviceType, Buffer> buffers;
    };

    DataType dtype = DataType::None;
    DeviceType currentDevice = DeviceType::None;
    std::shared_ptr<Storage> storage = std::make_shared<Storage>();
    Index identifier = 0;
    std::unordered_map<std::string, std::any> attributes;
    std::unordered_map<std::string, std::function<std::any()>> derivedAttributes;
    std::shared_ptr<Impl> attributeSource;

    struct Layout {
        Shape shape;
        Shape stride;
        Shape shapeMinusOne;
        Shape backstride;
        U64 offset = 0;
        U64 size = 0;
        U64 sizeBytes = 0;
        U64 offsetBytes = 0;
        U64 elementSize = 0;
        bool contiguous = true;
        bool initialized = false;

        Result computeDefaultStrides();
        Result updateCache();
        Result initialize(const Shape& shape, U64 elementSize);
        Result expand(Index axis);
        Result squeeze(Index axis);
        Result reshape(const Shape& newShape);
        Result broadcast(const Shape& targetShape);
        Result slice(const std::vector<Token>& slice);
        Result permute(const Shape& axes);
        U64 shapeToOffset(const std::initializer_list<U64>& coordinates) const;
    } layout;
};

Result Tensor::Impl::Layout::computeDefaultStrides() {
    const auto rank = shape.size();
    Shape candidate(rank);

    U64 strideVal = 1;
    for (std::size_t i = rank; i-- > 0;) {
        candidate[i] = strideVal;
        if (!detail::CheckedMultiply(strideVal, shape[i], strideVal)) {
            JST_ERROR("[MEMORY:TENSOR] Shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    stride = std::move(candidate);
    return Result::SUCCESS;
}

Result Tensor::Impl::Layout::updateCache() {
    const auto rank = shape.size();
    if (stride.size() != rank) {
        JST_ERROR("[MEMORY:TENSOR] Shape and stride ranks do not match.");
        return Result::ERROR;
    }

    if (!initialized) {
        if (rank != 0 || elementSize != 0 || offset != 0) {
            JST_ERROR("[MEMORY:TENSOR] Cannot cache a partially initialized layout.");
            return Result::ERROR;
        }

        shapeMinusOne.clear();
        backstride.clear();
        size = 0;
        sizeBytes = 0;
        offsetBytes = 0;
        contiguous = true;
        return Result::SUCCESS;
    }

    Shape candidateShapeMinusOne(rank);
    Shape candidateBackstride(rank);
    const bool hasZeroExtent = std::find(shape.begin(), shape.end(), 0) != shape.end();
    U64 candidateSize = hasZeroExtent ? 0 : 1;
    for (std::size_t i = 0; i < rank; ++i) {
        const U64 dim = shape[i];
        candidateShapeMinusOne[i] = dim > 0 ? dim - 1 : 0;
        if (!detail::CheckedMultiply(stride[i], candidateShapeMinusOne[i],
                                     candidateBackstride[i]) ||
            !detail::CheckedMultiply(candidateSize, dim, candidateSize)) {
            JST_ERROR("[MEMORY:TENSOR] Shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 candidateSizeBytes = 0;
    U64 candidateOffsetBytes = 0;
    if (!detail::CheckedMultiply(candidateSize, elementSize, candidateSizeBytes) ||
        !detail::CheckedMultiply(offset, elementSize, candidateOffsetBytes)) {
        JST_ERROR("[MEMORY:TENSOR] Tensor byte range exceeds the supported layout range.");
        return Result::ERROR;
    }

    bool candidateContiguous = true;
    U64 expectedStride = 1;
    for (std::size_t i = rank; i-- > 0;) {
        if (shape[i] == 1) {
            continue;
        }
        if (stride[i] != expectedStride) {
            candidateContiguous = false;
            break;
        }
        if (!detail::CheckedMultiply(expectedStride, shape[i], expectedStride)) {
            JST_ERROR("[MEMORY:TENSOR] Shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    shapeMinusOne = std::move(candidateShapeMinusOne);
    backstride = std::move(candidateBackstride);
    size = candidateSize;
    sizeBytes = candidateSizeBytes;
    offsetBytes = candidateOffsetBytes;
    contiguous = candidateContiguous;
    return Result::SUCCESS;
}

Result Tensor::Impl::Layout::initialize(const Shape& shape, U64 elementSize) {
    Layout candidate;
    candidate.shape = shape;
    candidate.elementSize = elementSize;
    candidate.initialized = true;
    JST_CHECK(candidate.computeDefaultStrides());
    JST_CHECK(candidate.updateCache());

    *this = std::move(candidate);
    return Result::SUCCESS;
}

Result Tensor::Impl::Layout::expand(Index axis) {
    if (axis > shape.size()) {
        JST_ERROR("[MEMORY:TENSOR] expand_dims axis {} out of range {}.", axis, shape.size());
        return Result::ERROR;
    }

    U64 newStride = 1;
    if (!stride.empty()) {
        if (axis == 0) {
            newStride = stride.front();
        } else {
            newStride = stride[axis - 1];
        }
    }

    shape.insert(shape.begin() + axis, 1);
    stride.insert(stride.begin() + axis, newStride);

    return updateCache();
}

Result Tensor::Impl::Layout::squeeze(Index axis) {
    if (axis >= shape.size()) {
        JST_ERROR("[MEMORY:TENSOR] squeeze_dims axis {} out of range {}.", axis, shape.size());
        return Result::ERROR;
    }

    if (shape[axis] != 1) {
        JST_ERROR("[MEMORY:TENSOR] Cannot squeeze dimension {} (size {}).", axis, shape[axis]);
        return Result::ERROR;
    }

    shape.erase(shape.begin() + axis);
    stride.erase(stride.begin() + axis);

    return updateCache();
}

Result Tensor::Impl::Layout::reshape(const Shape& newShape) {
    if (newShape.empty()) {
        JST_ERROR("[MEMORY:TENSOR] Cannot reshape to an empty shape.");
        return Result::ERROR;
    }

    if (!contiguous) {
        JST_ERROR("[MEMORY:TENSOR] Cannot reshape non-contiguous tensor.");
        return Result::ERROR;
    }

    const U64 originalSize = size;
    U64 newSize = 1;
    for (const auto dim : newShape) {
        if (dim == 0) {
            JST_ERROR("[MEMORY:TENSOR] Reshape dimension cannot be zero.");
            return Result::ERROR;
        }
        if (!detail::CheckedMultiply(newSize, dim, newSize)) {
            JST_ERROR("[MEMORY:TENSOR] Reshape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    if (originalSize != newSize) {
        JST_ERROR("[MEMORY:TENSOR] Reshape size mismatch {} -> {}.", originalSize, newSize);
        return Result::ERROR;
    }

    shape = newShape;
    JST_CHECK(computeDefaultStrides());
    return updateCache();
}

Result Tensor::Impl::Layout::broadcast(const Shape& targetShape) {
    if (targetShape.size() < shape.size()) {
        JST_ERROR("[MEMORY:TENSOR] Cannot broadcast shape {} -> {}.", shape, targetShape);
        return Result::ERROR;
    }

    const auto diff = targetShape.size() - shape.size();
    Shape paddedShape(targetShape.size(), 1);
    Shape paddedStride(targetShape.size(), stride.empty() ? 1 : stride.front());
    std::copy(shape.begin(), shape.end(), paddedShape.begin() + diff);
    std::copy(stride.begin(), stride.end(), paddedStride.begin() + diff);

    Shape newShape(targetShape.size());
    Shape newStride(targetShape.size());

    for (std::size_t i = 0; i < targetShape.size(); ++i) {
        const U64 currentDim = paddedShape[i];
        const U64 targetDim = targetShape[i];

        if (currentDim == targetDim) {
            newShape[i] = currentDim;
            newStride[i] = paddedStride[i];
        } else if (currentDim == 1) {
            newShape[i] = targetDim;
            newStride[i] = 0;
        } else {
            JST_ERROR("[MEMORY:TENSOR] Cannot broadcast dimension {} -> {} (axis {}).", currentDim, targetDim, i);
            return Result::ERROR;
        }
    }

    Layout candidate = *this;
    candidate.shape = std::move(newShape);
    candidate.stride = std::move(newStride);
    JST_CHECK(candidate.updateCache());

    *this = std::move(candidate);
    return Result::SUCCESS;
}

Result Tensor::Impl::Layout::slice(const std::vector<Token>& slice) {
    Shape newShape;
    Shape newStride;
    U64 offsetVal = offset;
    Index dim = 0;
    bool ellipsisUsed = false;

    for (const auto& token : slice) {
        switch (token.getType()) {
            case Token::Type::Number: {
                if (dim >= shape.size()) {
                    JST_ERROR("[MEMORY:TENSOR] Slice index exceeds dimensions.");
                    return Result::ERROR;
                }

                const U64 index = token.getA();
                if (index >= shape[dim]) {
                    JST_ERROR("[MEMORY:TENSOR] Slice index {} out of range {}.", index, shape[dim]);
                    return Result::ERROR;
                }

                U64 indexOffset = 0;
                if (!detail::CheckedMultiply(index, stride[dim], indexOffset) ||
                    !detail::CheckedAdd(offsetVal, indexOffset, offsetVal)) {
                    JST_ERROR("[MEMORY:TENSOR] Slice offset exceeds the supported layout range.");
                    return Result::ERROR;
                }
                dim++;
                break;
            }
            case Token::Type::Colon: {
                if (dim >= shape.size()) {
                    JST_ERROR("[MEMORY:TENSOR] Slice index exceeds dimensions.");
                    return Result::ERROR;
                }

                const U64 start = token.getA();
                U64 end = token.getB();
                const U64 step = token.getC();

                if (end == 0) {
                    end = shape[dim];
                }

                if (step == 0) {
                    JST_ERROR("[MEMORY:TENSOR] Slice step cannot be zero.");
                    return Result::ERROR;
                }

                if (start >= shape[dim] || end > shape[dim]) {
                    JST_ERROR("[MEMORY:TENSOR] Slice range [{}:{}] exceeds dimension {}.", start, end, shape[dim]);
                    return Result::ERROR;
                }

                if (start >= end) {
                    JST_ERROR("[MEMORY:TENSOR] Slice start must be less than end.");
                    return Result::ERROR;
                }

                const U64 range = end - start;
                newShape.push_back((range - 1) / step + 1);

                U64 slicedStride = 0;
                U64 startOffset = 0;
                if (!detail::CheckedMultiply(stride[dim], step, slicedStride) ||
                    !detail::CheckedMultiply(start, stride[dim], startOffset) ||
                    !detail::CheckedAdd(offsetVal, startOffset, offsetVal)) {
                    JST_ERROR("[MEMORY:TENSOR] Slice exceeds the supported layout range.");
                    return Result::ERROR;
                }
                newStride.push_back(slicedStride);
                dim++;
                break;
            }
            case Token::Type::Ellipsis: {
                if (ellipsisUsed) {
                    JST_ERROR("[MEMORY:TENSOR] Ellipsis can only appear once in slice.");
                    return Result::ERROR;
                }
                ellipsisUsed = true;

                const U64 remainingDims = shape.size() - (slice.size() - 1);
                while (dim < remainingDims) {
                    newShape.push_back(shape[dim]);
                    newStride.push_back(stride[dim]);
                    dim++;
                }
                break;
            }
        }
    }

    if (!ellipsisUsed) {
        while (dim < shape.size()) {
            newShape.push_back(shape[dim]);
            newStride.push_back(stride[dim]);
            dim++;
        }
    }

    shape = std::move(newShape);
    stride = std::move(newStride);
    offset = offsetVal;
    return updateCache();
}

Result Tensor::Impl::Layout::permute(const Shape& axes) {
    if (axes.empty()) {
        JST_ERROR("[MEMORY:TENSOR] Permutation cannot be empty.");
        return Result::ERROR;
    }

    if (axes.size() != shape.size()) {
        JST_ERROR("[MEMORY:TENSOR] Permutation rank {} does not match tensor rank {}.",
                  axes.size(), shape.size());
        return Result::ERROR;
    }

    Shape newShape(shape.size());
    Shape newStride(shape.size());
    std::vector<bool> seen(shape.size(), false);

    for (std::size_t i = 0; i < axes.size(); ++i) {
        const U64 axis = axes[i];

        if (axis >= shape.size()) {
            JST_ERROR("[MEMORY:TENSOR] Permutation axis {} out of range {}.",
                      axis, shape.size());
            return Result::ERROR;
        }

        if (seen[axis]) {
            JST_ERROR("[MEMORY:TENSOR] Permutation axis {} appears more than once.", axis);
            return Result::ERROR;
        }

        seen[axis] = true;
        newShape[i] = shape[axis];
        newStride[i] = stride[axis];
    }

    shape = std::move(newShape);
    stride = std::move(newStride);
    return updateCache();
}

U64 Tensor::Impl::Layout::shapeToOffset(const std::initializer_list<U64>& coordinates) const {
    if (shape.empty()) {
        return 0;
    }

    if (coordinates.size() < shape.size()) {
        JST_ERROR("[MEMORY:TENSOR] Coordinate rank {} smaller than tensor rank {}.", coordinates.size(), shape.size());
        return 0;
    }

    const std::size_t pad = coordinates.size() - shape.size();
    U64 index = 0;
    auto it = coordinates.begin();
    std::advance(it, pad);
    for (std::size_t i = 0; i < shape.size(); ++i, ++it) {
        index += (*it) * stride[i];
    }
    return index;
}

Tensor::Tensor() {
    ensureImpl();
}

Tensor::Tensor(void* pointer, const DeviceType& device, const DataType& dtype, const Shape& shape) {
    ensureImpl();
    auto status = create(pointer, device, dtype, shape);
    if (status != Result::SUCCESS) {
        JST_ERROR("[MEMORY:TENSOR] Failed to construct borrowed tensor {} on {}.", DataTypeToName(dtype), device);
    }
}

Tensor::Tensor(const DeviceType& device, const DataType& dtype, const Shape& shape) {
    ensureImpl();
    auto status = create(device, dtype, shape);
    if (status != Result::SUCCESS) {
        JST_ERROR("[MEMORY:TENSOR] Failed to construct tensor {} on {}.", DataTypeToName(dtype), device);
    }
}

Tensor::Tensor(const DeviceType& device, const Tensor& source) {
    ensureImpl();
    auto status = create(device, source);
    if (status != Result::SUCCESS) {
        JST_ERROR("[MEMORY:TENSOR] Failed to map tensor {} on {}.", DataTypeToName(source.dtype()), device);
    }
}

Tensor::~Tensor() = default;

void Tensor::ensureImpl() {
    if (!impl) {
        impl = std::make_shared<Tensor::Impl>();
    }
}

Result Tensor::create(const DeviceType& device, const DataType& dtype, const Shape& shape,
                      const Buffer::Config& config) {
    ensureImpl();

    if (dtype == DataType::None) {
        JST_ERROR("[MEMORY:TENSOR] Invalid data type.");
        return Result::ERROR;
    }

    if (shape.empty()) {
        JST_ERROR("[MEMORY:TENSOR] Shape cannot be empty.");
        return Result::ERROR;
    }

    Impl::Layout candidateLayout;
    JST_CHECK(candidateLayout.initialize(shape, DataTypeSize(dtype)));

    Buffer buffer;
    JST_CHECK(buffer.create(device, candidateLayout.sizeBytes, config));

    auto candidateStorage = std::make_shared<Impl::Storage>();
    candidateStorage->rootDevice = device;
    candidateStorage->buffers.emplace(device, std::move(buffer));

    impl->dtype = dtype;
    impl->layout = std::move(candidateLayout);
    impl->storage = std::move(candidateStorage);
    impl->currentDevice = device;
    impl->identifier = g_tensor_counter.fetch_add(1, std::memory_order_relaxed);

    return Result::SUCCESS;
}

Result Tensor::create(void* pointer, const DeviceType& device, const DataType& dtype, const Shape& shape) {
    ensureImpl();

    if (dtype == DataType::None) {
        JST_ERROR("[MEMORY:TENSOR] Invalid data type.");
        return Result::ERROR;
    }

    if (shape.empty()) {
        JST_ERROR("[MEMORY:TENSOR] Shape cannot be empty.");
        return Result::ERROR;
    }

    Impl::Layout candidateLayout;
    JST_CHECK(candidateLayout.initialize(shape, DataTypeSize(dtype)));

    if (impl->storage) {
        for (const auto& [_, existingBuffer] : impl->storage->buffers) {
            if (PointerRangesOverlap(pointer, candidateLayout.sizeBytes,
                                     existingBuffer.data(), existingBuffer.sizeBytes())) {
                JST_ERROR("[MEMORY:TENSOR] Cannot recreate a tensor by borrowing its current storage.");
                return Result::ERROR;
            }
        }
    }

    Buffer buffer;
    JST_CHECK(buffer.create(device, pointer, candidateLayout.sizeBytes));

    auto candidateStorage = std::make_shared<Impl::Storage>();
    candidateStorage->rootDevice = device;
    candidateStorage->buffers.emplace(device, std::move(buffer));

    impl->dtype = dtype;
    impl->layout = std::move(candidateLayout);
    impl->storage = std::move(candidateStorage);
    impl->currentDevice = device;
    impl->identifier = g_tensor_counter.fetch_add(1, std::memory_order_relaxed);

    return Result::SUCCESS;
}

Result Tensor::create(const DeviceType& device, const Tensor& source) {
    ensureImpl();

    if (!source.impl || impl == source.impl) {
        JST_ERROR("[MEMORY:TENSOR] Cannot map a tensor from itself or a value alias.");
        return Result::ERROR;
    }

    Buffer buffer;
    JST_CHECK(buffer.create(device, source.buffer()));

    auto candidateStorage = std::make_shared<Impl::Storage>();
    candidateStorage->rootDevice = source.nativeDevice();
    candidateStorage->buffers.emplace(device, std::move(buffer));
    candidateStorage->backingStorage = source.impl->storage;
    candidateStorage->backingOwners.push_back(source.buffer().ownershipToken());

    impl->dtype = source.dtype();
    impl->layout = source.impl->layout;
    impl->storage = std::move(candidateStorage);
    impl->currentDevice = device;
    impl->identifier = g_tensor_counter.fetch_add(1, std::memory_order_relaxed);
    impl->attributeSource = source.impl;

    return Result::SUCCESS;
}

Tensor Tensor::clone() const {
    Tensor copy;
    if (!impl) {
        return copy;
    }

    copy.ensureImpl();
    copy.impl->dtype = impl->dtype;
    copy.impl->currentDevice = impl->currentDevice;
    copy.impl->storage = impl->storage;
    copy.impl->identifier = g_tensor_counter.fetch_add(1, std::memory_order_relaxed);
    copy.impl->attributeSource = impl;

    copy.impl->layout = impl->layout;

    return copy;
}

const DeviceType& Tensor::device() const {
    static const DeviceType kNone = DeviceType::None;
    return impl ? impl->currentDevice : kNone;
}

const DeviceType& Tensor::nativeDevice() const {
    static const DeviceType kNone = DeviceType::None;
    return (impl && impl->storage) ? impl->storage->rootDevice : kNone;
}

const DataType& Tensor::dtype() const {
    static const DataType kNone = DataType::None;
    return impl ? impl->dtype : kNone;
}

const U64& Tensor::size() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.size : kZero;
}

const U64& Tensor::sizeBytes() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.sizeBytes : kZero;
}

const U64& Tensor::elementSize() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.elementSize : kZero;
}

bool Tensor::contiguous() const {
    return impl && impl->layout.contiguous;
}

bool Tensor::empty() const {
    return impl ? impl->layout.size == 0 : true;
}

bool Tensor::validShape() const {
    return impl && impl->layout.initialized;
}

const U64& Tensor::offset() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.offset : kZero;
}

const U64& Tensor::offsetBytes() const {
    static const U64 kZero = 0;
    return impl ? impl->layout.offsetBytes : kZero;
}

const Shape& Tensor::shape() const {
    static const Shape kEmpty{};
    return impl ? impl->layout.shape : kEmpty;
}

const Shape& Tensor::stride() const {
    static const Shape kEmpty{};
    return impl ? impl->layout.stride : kEmpty;
}

const Shape& Tensor::shapeMinusOne() const {
    static const Shape kEmpty{};
    return impl ? impl->layout.shapeMinusOne : kEmpty;
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

U64 Tensor::shapeToOffset(const std::initializer_list<U64>& coordinates) const {
    if (!impl) {
        return 0;
    }
    return impl->layout.shapeToOffset(coordinates);
}

Result Tensor::expandDims(Index axis) {
    if (!impl) {
        return Result::ERROR;
    }
    auto candidate = impl->layout;
    JST_CHECK(candidate.expand(axis));
    impl->layout = std::move(candidate);
    return Result::SUCCESS;
}

Result Tensor::squeezeDims(Index axis) {
    if (!impl) {
        return Result::ERROR;
    }
    auto candidate = impl->layout;
    JST_CHECK(candidate.squeeze(axis));
    impl->layout = std::move(candidate);
    return Result::SUCCESS;
}

Result Tensor::reshape(const Shape& newShape) {
    if (!impl) {
        return Result::ERROR;
    }
    auto candidate = impl->layout;
    JST_CHECK(candidate.reshape(newShape));
    impl->layout = std::move(candidate);
    return Result::SUCCESS;
}

Result Tensor::broadcastTo(const Shape& newShape) {
    if (!impl) {
        return Result::ERROR;
    }
    auto candidate = impl->layout;
    JST_CHECK(candidate.broadcast(newShape));
    impl->layout = std::move(candidate);
    return Result::SUCCESS;
}

Result Tensor::slice(const std::vector<Token>& tokens) {
    if (!impl) {
        return Result::ERROR;
    }
    auto candidate = impl->layout;
    JST_CHECK(candidate.slice(tokens));
    impl->layout = std::move(candidate);
    return Result::SUCCESS;
}

Result Tensor::permute(const Shape& axes) {
    if (!impl) {
        return Result::ERROR;
    }
    auto candidate = impl->layout;
    JST_CHECK(candidate.permute(axes));
    impl->layout = std::move(candidate);
    return Result::SUCCESS;
}

bool Tensor::hasDevice(const DeviceType& device) {
    ensureImpl();

    if (!impl->storage) {
        impl->storage = std::make_shared<Impl::Storage>();
    }

    if (impl->storage->buffers.contains(device)) {
        return true;
    }

    if (!impl->storage->buffers.contains(impl->storage->rootDevice)) {
        JST_ERROR("[MEMORY:TENSOR] Root buffer is not initialized.");
        return false;
    }

    Buffer clone;
    if (clone.create(device, impl->storage->buffers.at(impl->storage->rootDevice)) != Result::SUCCESS) {
        return false;
    }
    impl->storage->buffers.emplace(device, std::move(clone));

    return true;
}

Result Tensor::copyFrom(const Tensor& source, void* context) {
    if (!impl || !source.impl || !validShape() || !source.validShape() || dtype() == DataType::None || source.dtype() == DataType::None) {
        JST_ERROR("[MEMORY:TENSOR] Cannot copy uninitialized tensors.");
        return Result::ERROR;
    }

    if (shape() != source.shape()) {
        JST_ERROR("[MEMORY:TENSOR] Source tensor shape does not match destination tensor shape.");
        return Result::ERROR;
    }

    if (dtype() != source.dtype()) {
        JST_ERROR("[MEMORY:TENSOR] Source tensor dtype does not match destination tensor dtype.");
        return Result::ERROR;
    }

    if (size() != source.size() || sizeBytes() != source.sizeBytes()) {
        JST_ERROR("[MEMORY:TENSOR] Source tensor size does not match destination tensor size.");
        return Result::ERROR;
    }

    if (!source.contiguous()) {
        JST_ERROR("[MEMORY:TENSOR] Source tensor is not contiguous.");
        return Result::ERROR;
    }

    if (!contiguous()) {
        JST_ERROR("[MEMORY:TENSOR] Destination tensor is not contiguous.");
        return Result::ERROR;
    }

    if (source.offset() != 0 || offset() != 0) {
        JST_ERROR("[MEMORY:TENSOR] Cannot copy offset tensor views.");
        return Result::ERROR;
    }

    if (device() == DeviceType::None || source.device() == DeviceType::None || !impl->storage || !source.impl->storage) {
        JST_ERROR("[MEMORY:TENSOR] Cannot copy tensors without initialized storage.");
        return Result::ERROR;
    }

    const auto validBuffer = [](const std::shared_ptr<Impl::Storage>& storage, const DeviceType device, const U64 expectedSize) {
        const auto buffer = storage->buffers.find(device);
        return buffer != storage->buffers.end() && buffer->second.valid() && buffer->second.device() == device && buffer->second.sizeBytes() == expectedSize;
    };

    if (!validBuffer(source.impl->storage, source.device(), source.sizeBytes())) {
        JST_ERROR("[MEMORY:TENSOR] Source tensor buffer is invalid or does not cover its logical storage.");
        return Result::ERROR;
    }

    if (!validBuffer(impl->storage, device(), sizeBytes())) {
        JST_ERROR("[MEMORY:TENSOR] Destination tensor buffer is invalid or does not cover its logical storage.");
        return Result::ERROR;
    }

    const auto target = impl->storage->buffers.find(source.device());
    if (target != impl->storage->buffers.end() && (!target->second.valid() || target->second.device() != source.device() || target->second.sizeBytes() != sizeBytes())) {
        JST_ERROR("[MEMORY:TENSOR] Destination tensor mapping is invalid or does not cover its logical storage.");
        return Result::ERROR;
    }

    if (target == impl->storage->buffers.end() && !validBuffer(impl->storage, impl->storage->rootDevice, sizeBytes())) {
        JST_ERROR("[MEMORY:TENSOR] Destination tensor root buffer cannot provide the requested mapping.");
        return Result::ERROR;
    }

    if (impl->storage == source.impl->storage) {
        return Result::SUCCESS;
    }

    // Try to map the buffer to the source device.
    if (!hasDevice(source.device())) {
        JST_ERROR("[MEMORY:TENSOR] Source tensor device is not supported.");
        return Result::ERROR;
    }

    // Copy the buffer from the source device to the current device.
    JST_CHECK(impl->storage->buffers.at(source.device()).copyFrom(source.impl->storage->buffers.at(source.device()), context));

    return Result::SUCCESS;
}

Result Tensor::swapBuffers(Tensor& other) {
    ensureImpl();
    other.ensureImpl();

    if (this == &other || impl == other.impl) {
        return Result::SUCCESS;
    }

    if (!impl->storage || !other.impl->storage) {
        JST_ERROR("[MEMORY:TENSOR] Cannot swap tensor storage before initialization.");
        return Result::ERROR;
    }

    if (impl->storage == other.impl->storage) {
        return Result::SUCCESS;
    }

    if (dtype() == DataType::None || other.dtype() == DataType::None) {
        JST_ERROR("[MEMORY:TENSOR] Cannot swap buffers for uninitialized tensors.");
        return Result::ERROR;
    }

    if (dtype() != other.dtype()) {
        JST_ERROR("[MEMORY:TENSOR] Cannot swap buffers with mismatched dtypes ({} vs {}).",
                  dtype(), other.dtype());
        return Result::ERROR;
    }

    if (shape() != other.shape() ||
        stride() != other.stride() ||
        offset() != other.offset() ||
        sizeBytes() != other.sizeBytes()) {
        JST_ERROR("[MEMORY:TENSOR] Cannot swap buffers for tensors with different layouts.");
        return Result::ERROR;
    }

    if (device() == DeviceType::None || other.device() == DeviceType::None) {
        JST_ERROR("[MEMORY:TENSOR] Cannot swap buffers without an active device.");
        return Result::ERROR;
    }

    if (device() != other.device()) {
        JST_ERROR("[MEMORY:TENSOR] Cannot swap buffers across active devices ({} vs {}).",
                  device(), other.device());
        return Result::ERROR;
    }

    if (nativeDevice() != other.nativeDevice()) {
        JST_ERROR("[MEMORY:TENSOR] Cannot swap buffers across native devices ({} vs {}).",
                  nativeDevice(), other.nativeDevice());
        return Result::ERROR;
    }

    const auto& lhsBuffers = impl->storage->buffers;
    const auto& rhsBuffers = other.impl->storage->buffers;
    if (lhsBuffers.size() != rhsBuffers.size()) {
        JST_ERROR("[MEMORY:TENSOR] Cannot swap buffers with different materialized device sets.");
        return Result::ERROR;
    }

    for (const auto& [deviceKey, _] : lhsBuffers) {
        if (!rhsBuffers.contains(deviceKey)) {
            JST_ERROR("[MEMORY:TENSOR] Cannot swap buffers with different materialized device sets.");
            return Result::ERROR;
        }
    }

    std::swap(impl->storage->rootDevice, other.impl->storage->rootDevice);
    std::swap(impl->storage->buffers, other.impl->storage->buffers);

    return Result::SUCCESS;
}

void* Tensor::data() {
    if (!impl || impl->currentDevice == DeviceType::None) {
        return nullptr;
    }

    if (!impl->storage || !impl->storage->buffers.contains(impl->currentDevice)) {
        return nullptr;
    }

    auto& buf = impl->storage->buffers.at(impl->currentDevice);
    void* base = buf.data();
    if (!base) {
        return nullptr;
    }

    if (impl->currentDevice == DeviceType::CPU) {
        auto* bytes = static_cast<std::uint8_t*>(base);
        return bytes + impl->layout.offsetBytes;
    }

    return base;
}

const void* Tensor::data() const {
    if (!impl || impl->currentDevice == DeviceType::None) {
        return nullptr;
    }

    if (!impl->storage || !impl->storage->buffers.contains(impl->currentDevice)) {
        return nullptr;
    }

    const auto& buf = impl->storage->buffers.at(impl->currentDevice);
    const void* base = buf.data();
    if (!base) {
        return nullptr;
    }

    if (impl->currentDevice == DeviceType::CPU) {
        auto* bytes = static_cast<const std::uint8_t*>(base);
        return bytes + impl->layout.offsetBytes;
    }

    return base;
}

const Buffer& Tensor::buffer() const {
    if (!impl || impl->currentDevice == DeviceType::None) {
        throw std::runtime_error("Tensor buffer requested before initialization");
    }
    if (!impl->storage) {
        JST_ERROR("[MEMORY:TENSOR] Buffer storage not initialized.");
        throw std::runtime_error("Tensor buffer storage not initialized");
    }

    const auto it = impl->storage->buffers.find(impl->currentDevice);
    if (it == impl->storage->buffers.end()) {
        JST_ERROR("[MEMORY:TENSOR] Buffer for device {} is not materialized.", impl->currentDevice);
        throw std::runtime_error("Tensor buffer not materialized for device");
    }
    return it->second;
}

Buffer& Tensor::buffer() {
    if (!impl || impl->currentDevice == DeviceType::None) {
        throw std::runtime_error("Tensor buffer requested before initialization");
    }
    if (!impl->storage) {
        JST_ERROR("[MEMORY:TENSOR] Buffer storage not initialized.");
        throw std::runtime_error("Tensor buffer storage not initialized");
    }

    const auto it = impl->storage->buffers.find(impl->currentDevice);
    if (it == impl->storage->buffers.end()) {
        JST_ERROR("[MEMORY:TENSOR] Buffer for device {} is not materialized.", impl->currentDevice);
        throw std::runtime_error("Tensor buffer not materialized for device");
    }
    return it->second;
}

const Index& Tensor::id() const {
    return impl->identifier;
}

bool Tensor::hasAttribute(const std::string& key) const {
    if (!impl) {
        return false;
    }

    const Impl* current = impl.get();
    while (current) {
        if (current->derivedAttributes.contains(key)) {
            return true;
        }
        if (current->attributes.contains(key)) {
            return true;
        }
        current = current->attributeSource.get();
    }

    return false;
}

std::vector<std::string> Tensor::attributeKeys() const {
    std::vector<std::string> keys;

    if (!impl) {
        return keys;
    }

    const Impl* current = impl.get();
    while (current) {
        for (const auto& [key, _] : current->derivedAttributes) {
            if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
                keys.push_back(key);
            }
        }

        for (const auto& [key, _] : current->attributes) {
            if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
                keys.push_back(key);
            }
        }

        current = current->attributeSource.get();
    }

    std::sort(keys.begin(), keys.end());

    return keys;
}

Result Tensor::setAttribute(const std::string& key, const std::any& value) {
    if (!impl) {
        JST_ERROR("[MEMORY:TENSOR] Tensor not initialized.");
        return Result::ERROR;
    }

    impl->attributes[key] = value;

    return Result::SUCCESS;
}

Result Tensor::setDerivedAttribute(const std::string& key,
                                   std::function<std::any()> compute) {
    if (!impl) {
        JST_ERROR("[MEMORY:TENSOR] Tensor not initialized.");
        return Result::ERROR;
    }

    impl->derivedAttributes[key] = std::move(compute);

    return Result::SUCCESS;
}

std::any Tensor::attribute(const std::string& key) const {
    if (!impl) {
        JST_ERROR("[MEMORY:TENSOR] Tensor not initialized.");
        return std::any();
    }

    const Impl* current = impl.get();
    while (current) {
        auto dit = current->derivedAttributes.find(key);
        if (dit != current->derivedAttributes.end()) {
            return dit->second();
        }
        auto ait = current->attributes.find(key);
        if (ait != current->attributes.end()) {
            return ait->second;
        }
        current = current->attributeSource.get();
    }

    JST_ERROR("[MEMORY:TENSOR] Attribute '{}' does not exist.", key);
    return std::any();
}

Result Tensor::propagateAttributes(const Tensor& source) {
    ensureImpl();
    // Skip if source and destination share the same impl (e.g. view
    // modules like expand_dims/squeeze_dims/reshape). The attributes
    // are already accessible since the impl is identical.
    if (impl != source.impl) {
        impl->attributeSource = source.impl;
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream
