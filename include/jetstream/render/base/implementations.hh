#ifndef JETSTREAM_RENDER_BASE_IMPLEMENTATIONS_HH
#define JETSTREAM_RENDER_BASE_IMPLEMENTATIONS_HH

#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<DeviceType D> class BufferImp;
template<DeviceType D> class ProgramImp;
template<DeviceType D> class KernelImp;
template<DeviceType D> class DrawImp;
template<DeviceType D> class SurfaceImp;
template<DeviceType D> class TextureImp;
template<DeviceType D> class VertexImp;
template<DeviceType D> class WindowImp;

}  // namespace Jetstream::Render

#endif
