#ifndef JETSTREAM_RENDER_EXTRAS_HH
#define JETSTREAM_RENDER_EXTRAS_HH

#include <vector>

#include "jetstream/types.hh"

namespace Jetstream::Render::Extras {

inline float FillScreenVertices[] = {
    +1.0f, -1.0f, 0.0f,
    +1.0f, +1.0f, 0.0f,
    -1.0f, +1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f,
};

inline float FillScreenTextureVertices[] = {
    +1.0f, +1.0f,
    +1.0f, +0.0f,
    +0.0f, +0.0f,
    +0.0f, +1.0f,
};

inline uint32_t FillScreenIndices[] = {
    0, 1, 2,
    2, 3, 0,
};

// Copyright 2019 Google LLC.
// SPDX-License-Identifier: Apache-2.0
// Author: Anton Mikhailov
const unsigned char TurboLutBytes[256][4] = {{0,0,0,255},{50,21,67,255},{51,24,74,255},{52,27,81,255},{53,30,88,255},{54,33,95,255},{55,36,102,255},{56,39,109,255},{57,42,115,255},{58,45,121,255},{59,47,128,255},{60,50,134,255},{61,53,139,255},{62,56,145,255},{63,59,151,255},{63,62,156,255},{64,64,162,255},{65,67,167,255},{65,70,172,255},{66,73,177,255},{66,75,181,255},{67,78,186,255},{68,81,191,255},{68,84,195,255},{68,86,199,255},{69,89,203,255},{69,92,207,255},{69,94,211,255},{70,97,214,255},{70,100,218,255},{70,102,221,255},{70,105,224,255},{70,107,227,255},{71,110,230,255},{71,113,233,255},{71,115,235,255},{71,118,238,255},{71,120,240,255},{71,123,242,255},{70,125,244,255},{70,128,246,255},{70,130,248,255},{70,133,250,255},{70,135,251,255},{69,138,252,255},{69,140,253,255},{68,143,254,255},{67,145,254,255},{66,148,255,255},{65,150,255,255},{64,153,255,255},{62,155,254,255},{61,158,254,255},{59,160,253,255},{58,163,252,255},{56,165,251,255},{55,168,250,255},{53,171,248,255},{51,173,247,255},{49,175,245,255},{47,178,244,255},{46,180,242,255},{44,183,240,255},{42,185,238,255},{40,188,235,255},{39,190,233,255},{37,192,231,255},{35,195,228,255},{34,197,226,255},{32,199,223,255},{31,201,221,255},{30,203,218,255},{28,205,216,255},{27,208,213,255},{26,210,210,255},{26,212,208,255},{25,213,205,255},{24,215,202,255},{24,217,200,255},{24,219,197,255},{24,221,194,255},{24,222,192,255},{24,224,189,255},{25,226,187,255},{25,227,185,255},{26,228,182,255},{28,230,180,255},{29,231,178,255},{31,233,175,255},{32,234,172,255},{34,235,170,255},{37,236,167,255},{39,238,164,255},{42,239,161,255},{44,240,158,255},{47,241,155,255},{50,242,152,255},{53,243,148,255},{56,244,145,255},{60,245,142,255},{63,246,138,255},{67,247,135,255},{70,248,132,255},{74,248,128,255},{78,249,125,255},{82,250,122,255},{85,250,118,255},{89,251,115,255},{93,252,111,255},{97,252,108,255},{101,253,105,255},{105,253,102,255},{109,254,98,255},{113,254,95,255},{117,254,92,255},{121,254,89,255},{125,255,86,255},{128,255,83,255},{132,255,81,255},{136,255,78,255},{139,255,75,255},{143,255,73,255},{146,255,71,255},{150,254,68,255},{153,254,66,255},{156,254,64,255},{159,253,63,255},{161,253,61,255},{164,252,60,255},{167,252,58,255},{169,251,57,255},{172,251,56,255},{175,250,55,255},{177,249,54,255},{180,248,54,255},{183,247,53,255},{185,246,53,255},{188,245,52,255},{190,244,52,255},{193,243,52,255},{195,241,52,255},{198,240,52,255},{200,239,52,255},{203,237,52,255},{205,236,52,255},{208,234,52,255},{210,233,53,255},{212,231,53,255},{215,229,53,255},{217,228,54,255},{219,226,54,255},{221,224,55,255},{223,223,55,255},{225,221,55,255},{227,219,56,255},{229,217,56,255},{231,215,57,255},{233,213,57,255},{235,211,57,255},{236,209,58,255},{238,207,58,255},{239,205,58,255},{241,203,58,255},{242,201,58,255},{244,199,58,255},{245,197,58,255},{246,195,58,255},{247,193,58,255},{248,190,57,255},{249,188,57,255},{250,186,57,255},{251,184,56,255},{251,182,55,255},{252,179,54,255},{252,177,54,255},{253,174,53,255},{253,172,52,255},{254,169,51,255},{254,167,50,255},{254,164,49,255},{254,161,48,255},{254,158,47,255},{254,155,45,255},{254,153,44,255},{254,150,43,255},{254,147,42,255},{254,144,41,255},{253,141,39,255},{253,138,38,255},{252,135,37,255},{252,132,35,255},{251,129,34,255},{251,126,33,255},{250,123,31,255},{249,120,30,255},{249,117,29,255},{248,114,28,255},{247,111,26,255},{246,108,25,255},{245,105,24,255},{244,102,23,255},{243,99,21,255},{242,96,20,255},{241,93,19,255},{240,91,18,255},{239,88,17,255},{237,85,16,255},{236,83,15,255},{235,80,14,255},{234,78,13,255},{232,75,12,255},{231,73,12,255},{229,71,11,255},{228,69,10,255},{226,67,10,255},{225,65,9,255},{223,63,8,255},{221,61,8,255},{220,59,7,255},{218,57,7,255},{216,55,6,255},{214,53,6,255},{212,51,5,255},{210,49,5,255},{208,47,5,255},{206,45,4,255},{204,43,4,255},{202,42,4,255},{200,40,3,255},{197,38,3,255},{195,37,3,255},{193,35,2,255},{190,33,2,255},{188,32,2,255},{185,30,2,255},{183,29,2,255},{180,27,1,255},{178,26,1,255},{175,24,1,255},{172,23,1,255},{169,22,1,255},{167,20,1,255},{164,19,1,255},{161,18,1,255},{158,16,1,255},{155,15,1,255},{152,14,1,255},{149,13,1,255},{146,11,1,255},{142,10,1,255},{139,9,2,255},{136,8,2,255},{133,7,2,255},{129,6,2,255},{126,5,2,255},{122,4,3,255}};

}  // namespace Jetstream::Render::Extras

#endif
