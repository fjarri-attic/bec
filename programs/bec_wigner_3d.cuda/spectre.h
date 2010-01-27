#ifndef _SPECTRE_H
#define _SPECTRE_H

#include "cudabuffer.h"
#include "cudatexture.h"

void initSpectre();
void releaseSpectre();

// draw heightmap from data to canvas
// max is the height that will correspond to the maximum color of the spectre
void drawData(CudaTexture &canvas, CudaBuffer<value_type> &data, value_type max);

#endif
