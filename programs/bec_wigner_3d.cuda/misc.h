#ifndef _MISC_H
#define _MISC_H

#include <cutil_inline.h>

void createKernelParams(dim3 &block, dim3 &grid, int size, int block_size);

#endif
