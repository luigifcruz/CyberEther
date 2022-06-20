#ifndef JETSTREAM_RENDER_BASE_IMPLEMENTATIONS_HH
#define JETSTREAM_RENDER_BASE_IMPLEMENTATIONS_HH

#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<Device D> class BufferImp;
template<Device D> class ProgramImp;
template<Device D> class DrawImp;
template<Device D> class SurfaceImp;
template<Device D> class TextureImp;
template<Device D> class VertexImp;
template<Device D> class WindowImp;

}  // namespace Jetstream::Render

#endif
