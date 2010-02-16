#ifndef _REDUCE_CUH
#define _REDUCE_CUH

#include <cutil_inline.h>

#ifndef MAX_GRID_SIZE
#define MAX_GRID_SIZE 32768
#endif

inline value_pair operator*(value_pair a, value_pair b) { return MAKE_VALUE_PAIR(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
inline value_pair operator*(value_pair a, value_type b) { return MAKE_VALUE_PAIR(b*a.x, b*a.y); }
inline value_pair operator/(value_pair a, value_type b) { return MAKE_VALUE_PAIR(a.x / b, a.y / b); }
inline value_pair operator+(value_pair a, value_pair b) { return MAKE_VALUE_PAIR(a.x + b.x, a.y + b.y ); }
inline value_pair operator-(value_pair a, value_pair b) { return MAKE_VALUE_PAIR(a.x - b.x, a.y - b.y ); }
inline void operator+=(value_pair& a, const value_pair& b) { a.x += b.x; a.y += b.y; }
inline value_type abs(value_pair a) { return a.x * a.x + a.y * a.y; }

/**
 * Reduction kernel
 *
 * Parameterized by value type and number of threads in block (it is used to increase performance)
 * Reduction power equals to twice the number of threads in block
 *
 * @param output Output buffer
 * @param input Input buffer
 */
template <class T, unsigned int blockSize>
__global__ void reduceKernel(T* output, const T* input)
{
	extern __shared__ T shared_mem[];

	int tid = threadIdx.x;
	int bid = blockIdx.y * gridDim.x + blockIdx.x;

	// first reduction, after which the number of elements to reduce
	// equals to number of threads in block
	shared_mem[tid] = input[tid + 2 * bid * blockSize] +
		input[tid + 2 * bid * blockSize + blockSize];

	__syncthreads();

	// 'if(blockSize)'s will be removed by preprocessor, because blockSize is known in compile-time
	// 'if(tid)'s will not, but since they split execution only near the border of warps,
	// they are not affecting performance (i.e, for each warp there will be only one path
	// of execution anyway)
	if (blockSize >= 512) { if (tid < 256) { shared_mem[tid] += shared_mem[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { shared_mem[tid] += shared_mem[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { shared_mem[tid] += shared_mem[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) shared_mem[tid] += shared_mem[tid + 32];
		if (blockSize >= 32) shared_mem[tid] += shared_mem[tid + 16];
		if (blockSize >= 16) shared_mem[tid] += shared_mem[tid + 8];
		if (blockSize >= 8) shared_mem[tid] += shared_mem[tid + 4];
		if (blockSize >= 4) shared_mem[tid] += shared_mem[tid + 2];
		if (blockSize >= 2) shared_mem[tid] += shared_mem[tid + 1];
	}

	if (tid == 0) output[bid] = shared_mem[0];
}


/**
 * Reduce the array of values to a given extent. Neighbouring values are reduced first
 * (i.e., (1 2 3 4) will be reduced to (1+2 3+4) and then to (1+2+3+4))
 * Operation is in-place and can spoil the initial data.
 *
 * @param d_data Buffer with data, in device memory
 * @param d_temp Temporary buffer, should have at least length of (length(d_data) / min(1024, length/final_length))
 * @param length Length of d_data buffer
 * @param final_length Length of reduced data
 *
 * @return Reduction result if final_length==1, otherwise 0 and leaves reduced array in d_data
 */
template<class T>
T reduce(T* d_data, T* d_temp, int length, int final_length = 1)
{
	int reduce_power;

	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);

	int shared_mem_size;

	T *d_input = d_data;
	T *d_output = d_temp;
	T *temp;

	// reduce until length is not equal to final length
	while(length > final_length)
	{
		// we can reduce maximum 1024 (maximum block size * 2) times a pass
		reduce_power = (length / final_length >= 1024) ? 1024 : length / final_length;

		grid.x = (length >= reduce_power) ? (length / reduce_power) : 1;
		if(grid.x > MAX_GRID_SIZE)
		{
			grid.y = grid.x / MAX_GRID_SIZE;
			grid.x = MAX_GRID_SIZE;
		}
		else
			grid.y = 1;

		block.x = (length >= reduce_power) ? reduce_power / 2 : length / 2;

		shared_mem_size = (block.x + ((block.x > 32) ? 0 : block.x / 2)) * sizeof(T);

		length /= reduce_power;

		switch(reduce_power)
		{
		case 1024:
			reduceKernel<T, 512><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 512:
			reduceKernel<T, 256><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 256:
			reduceKernel<T, 128><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 128:
			reduceKernel<T, 64><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 64:
			reduceKernel<T, 32><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 32:
			reduceKernel<T, 16><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 16:
			reduceKernel<T, 8><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 8:
			reduceKernel<T, 4><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 4:
			reduceKernel<T, 2><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		case 2:
			reduceKernel<T, 1><<<grid, block, shared_mem_size>>>(d_output, d_input); break;
		}
		cutilCheckMsg("reduceKernel");

		// swap buffers
		temp = d_input;
		d_input = d_output;
		d_output = temp;
	}

	if(final_length == 1)
	// return reduction result
	{
		T h_result[1];
		cutilSafeCall(cudaMemcpy(h_result, d_input, sizeof(T), cudaMemcpyDeviceToHost));
		return h_result[0];
	}
	else
	// copy reduced array to the initial buffer
	{
		if(d_input != d_data)
			cutilSafeCall(cudaMemcpy(d_data, d_input, sizeof(T) * final_length, cudaMemcpyDeviceToDevice));
		return T();
	}
}

#endif
