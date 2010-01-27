#include <cutil_inline.h>
#include "defines.h"
#include "misc.h"
#include "spectre.h"

// texture with mapping level (0.0..1.0) -> color
texture<float4, 1> spectre_tex;
cudaArray *spectre_arr;

// returns black for index < 0, red for index > 255 * 6 and
// violet-blue-teal-green-yellow-red for intermediate indices
dim3 violetToRedSpectre(int index)
{
	if(index < 0)
		return dim3(0, 0, 0);
	else if(index <= 255) // black -> violet
		return dim3(index, 0, index);
	else if(index <= 255 * 2) // violet -> blue
		return dim3(255 * 2 - index, 0, 255);
	else if(index <= 255 * 3) // blue -> teal
		return dim3(0, index - 255 * 2, 255);
	else if(index <= 255 * 4) // teal -> green
		return dim3(0, 255, 255 * 4 - index);
	else if(index <= 255 * 5) // green -> yellow
		return dim3(index - 255 * 4, 255, 0);
	else if(index <= 255 * 6) // yellow -> red
		return dim3(255, 255 * 6 - index, 0);
	else
		return dim3(255, 0, 0);
}

// returns blue for index < 0, red for index > 255 * 3 and
// blue-white-yellow-red for intermediate indices
dim3 blueToRedSpectre(int index)
{
	if(index < 0)
		return dim3(0, 0, 0);
	else if(index <= 255) // blue -> white
		return dim3(index, index, 255);
	else if(index <= 255 * 2) // white -> yellow
		return dim3(255, 255, 255 * 2 - index);
	else if(index <= 255 * 3) // yellow -> red
		return dim3(255, 255 * 3 - index, 0);
	else
		return dim3(255, 0, 0);
}

// fills array with spectre colors using given index->color function
void fillSpectreArray(float4 *array, dim3 (*func)(int), int colors_num)
{
	for(int i = 0; i < colors_num; i++)
	{
		dim3 color = func(i);
		array[i].x = (float)color.x / 256;
		array[i].y = (float)color.y / 256;
		array[i].z = (float)color.z / 256;
		array[i].w = 1.0;
	}
}

// Prepare 'rainbow' texture which is used to display 2D heightmaps
void initSpectre()
{
	// int colors_num = 255 * 6;
	int colors_num = 255 * 3;

	float4 *h_colors = new float4[colors_num];

	// fillSpectreArray(h_colors, violetToRedSpectre, colors_num);
	fillSpectreArray(h_colors, blueToRedSpectre, colors_num);

	// texture parameters - linear filtering, value range [0.0, 1.0],
	// f(x) = f(0) for x < 0 and f(x) = f(1) for x > 1
	spectre_tex.filterMode = cudaFilterModeLinear;
	spectre_tex.normalized = true;
	spectre_tex.addressMode[0] = cudaAddressModeClamp;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();

	cutilSafeCall(cudaMallocArray(&spectre_arr, &desc, colors_num, 1));
	cutilSafeCall(cudaMemcpyToArray(spectre_arr, 0, 0, h_colors, colors_num * sizeof(float4), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaBindTextureToArray(spectre_tex, spectre_arr, desc));

	delete[] h_colors;
}

// Clean up coloring texture
void releaseSpectre()
{
	cudaUnbindTexture(spectre_tex);
	cudaFreeArray(spectre_arr);
}

// Kernel for transforming levels to colors
__global__ void drawKernel(float4 *canvas, value_type *data, value_type max)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	canvas[index] = tex1D(spectre_tex, data[index] / max);
}

void drawData(CudaTexture &canvas, CudaBuffer<value_type> &data, value_type max)
{
	dim3 grid, block;
	createKernelParams(block, grid, data.len(), MAX_THREADS_NUM);

	float4 *canvas_ptr = canvas.map();

	drawKernel<<<grid, block>>>(canvas_ptr, data, max);
	cutilCheckMsg("drawKernel");

	canvas.unmap();
}
