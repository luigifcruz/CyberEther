#ifndef JETSTREAM_RENDER_TOOLS_STB_DECOMPRESS_HH
#define JETSTREAM_RENDER_TOOLS_STB_DECOMPRESS_HH

namespace Jetstream::Render::Tools {

unsigned int stb_decompress_length(const unsigned char* input);
unsigned int stb_decompress(unsigned char* output, const unsigned char* input, unsigned int length);

}  // namespace Jetstream::Render::Tools

#endif  // JETSTREAM_RENDER_TOOLS_STB_DECOMPRESS_HH
