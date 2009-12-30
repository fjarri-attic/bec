#include "misc.h"
#include <cutil_inline.h>

#ifndef MAX_GRID_SIZE
#define MAX_GRID_SIZE 32768
#endif

void createKernelParams(dim3 &block, dim3 &grid, int size, int block_size)
{
	block.x = size > block_size ? block_size : size;
	grid.x = size / block.x;
	if(grid.x > MAX_GRID_SIZE)
	{
		grid.y = grid.x / MAX_GRID_SIZE;
		grid.x = MAX_GRID_SIZE;
	}
	else
		grid.y = 1;

	grid.z = 1;
}
