#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include <math.h>

#include "defines.h"

// texture with mapping level -> color
// ([0.0, 1.0] -> [black-violet-blue-green-yellow-red])
texture<float4, 1> rainbow_tex;
cudaArray *rainbow_arr;

texture<value_type, 1> k_tex;
cudaArray *k_arr;
value_type *k_buf;

// calculation parameters in constant memory, for easy access from kernels
__constant__ CalculationParameters d_params;

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
void setupRainbow()
{
	// int colors_num = 255 * 6;
	int colors_num = 255 * 3;

	float4 *h_colors = new float4[colors_num];

	// fillSpectreArray(h_colors, violetToRedSpectre, colors_num);
	fillSpectreArray(h_colors, blueToRedSpectre, colors_num);

	// texture parameters - linear filtering, value range [0.0, 1.0],
	// f(x) = f(0) for x < 0 and f(x) = f(1) for x > 1
	rainbow_tex.filterMode = cudaFilterModeLinear;
	rainbow_tex.normalized = true;
	rainbow_tex.addressMode[0] = cudaAddressModeClamp;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();

	cutilSafeCall(cudaMallocArray(&rainbow_arr, &desc, colors_num, 1));
	cutilSafeCall(cudaMemcpyToArray(rainbow_arr, 0, 0, h_colors, colors_num * sizeof(float4), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaBindTextureToArray(rainbow_tex, rainbow_arr, desc));

	delete[] h_colors;
}

// Clean up coloring texture
void deleteRainbow()
{
	cudaUnbindTexture(rainbow_tex);
	cudaFreeArray(rainbow_arr);
}

// returns color for level in range [0.0, 1.0]
__device__ __inline__ float4 getRainbowColor(float normalized)
{
	return tex1D(rainbow_tex, normalized);
}

value_type h_kkk(int i, value_type dk, int N)
{
//	value_type alpha = (N - 3.0) / (N - 2.0);
//	value_type beta = (N - 1.0) / 2;

//	if(2 * i >= N)
//		return dk / (beta - alpha * (N - 1 - i));
//	else
//		return dk / (beta - alpha * i);
	if(2 * i > N)
		return dk * (i - N);
	else
		return dk * i;
}

value_type h_kkk2(int i, value_type dk, int N)
{
	return (2 * i > N) ? (dk * (i - N)) : (dk * i);
}

void setupK(CalculationParameters &params)
{
	value_type *h_k = new value_type[params.cells];

	for(int index = 0; index < params.cells; index++)
	{
		int k = index >> (params.nvx_pow + params.nvy_pow);
		int k_shift = (k << (params.nvx_pow + params.nvy_pow));
		int j = (index - k_shift) >> params.nvx_pow;
		int i = (index - k_shift) - (j << params.nvx_pow);

		value_type kx = h_kkk(i, params.dkx, params.nvx);
		value_type ky = h_kkk(j, params.dky, params.nvy);
		value_type kz = h_kkk(k, params.dkz, params.nvz);

		h_k[index] = (kx * kx + ky * ky + kz * kz) * params.kcoeff;
	}

	k_tex.filterMode = cudaFilterModeLinear;
	k_tex.normalized = false;
	k_tex.addressMode[0] = cudaAddressModeClamp;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<value_type>();

	// not using array, because number of elements can be rather big
	cutilSafeCall(cudaMalloc((void**)&k_buf, sizeof(value_type) * params.cells));
	cutilSafeCall(cudaMemcpy(k_buf, h_k, sizeof(value_type) * params.cells, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaBindTexture(0, k_tex, k_buf, desc, sizeof(value_type) * params.cells));

	delete[] h_k;
}

void deleteK()
{
	cudaUnbindTexture(k_tex);
	cudaFreeArray(k_arr);
}

void setupTextures(CalculationParameters &params)
{
	setupRainbow();
	setupK(params);
}

void deleteTextures()
{
	deleteRainbow();
	deleteK();
}

__device__ value_type kkk(int i, value_type dk, int N)
{
	value_type alpha = (N - 3.0) / (N - 2.0);
	value_type beta = (N - 1.0) / 2;

	if(2 * i >= N)
		return dk / (beta - alpha * (N - 1 - i));
	else
		return dk / (beta - alpha * i);
}

// Returns squared module of k vector
__device__ value_type kradius(int index)
{
	int nvx_pow = d_params.nvx_pow;
	int nvy_pow = d_params.nvy_pow;
	int k = index >> (nvx_pow + nvy_pow);
	index -= (k << (nvx_pow + nvy_pow));
	int j = index >> nvx_pow;
	int i = index - (j << nvx_pow);

//	value_type kx = (2 * i > d_params.nvx) ? (d_params.dkx * (i - d_params.nvx)) : (d_params.dkx * i);
//	value_type ky = (2 * j > d_params.nvy) ? (d_params.dky * (j - d_params.nvy)) : (d_params.dky * j);
//	value_type kz = (2 * k > d_params.nvz) ? (d_params.dkz * (k - d_params.nvz)) : (d_params.dkz * k);
	value_type kx = kkk(i, d_params.dkx, d_params.nvx);
	value_type ky = kkk(j, d_params.dky, d_params.nvy);
	value_type kz = kkk(k, d_params.dkz, d_params.nvz);

	return kx * kx + ky * ky + kz * kz;
}


__device__ __inline__ value_pair cmul(value_pair a, value_pair b)
{
	return MAKE_VALUE_PAIR(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __inline__ value_pair cadd(value_pair a, value_pair b)
{
	return MAKE_VALUE_PAIR(a.x + b.x, a.y + b.y);
}

__device__ __inline__ value_pair cmul(value_pair a, value_type b)
{
	return MAKE_VALUE_PAIR(a.x * b, a.y * b);
}

// Returns potential energy
__device__ __inline__ value_type potential(int index)
{
	int nvx_pow = d_params.nvx_pow;
	int nvy_pow = d_params.nvy_pow;
	int k = index >> (nvx_pow + nvy_pow);
	index -= (k << (nvx_pow + nvy_pow));
	int j = index >> nvx_pow;
	int i = index - (j << nvx_pow);

	value_type x = -d_params.xmax + d_params.dx * i;
	value_type y = -d_params.ymax + d_params.dy * j;
	value_type z = -d_params.zmax + d_params.dz * k;

	return d_params.px * x * x + d_params.py * y * y + d_params.pz * z * z;
}

__global__ void fillWithTFSolution(value_pair *data)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_type e = d_params.mu - potential(index) / 2;
	if(e > 0)
		data[index] = MAKE_VALUE_PAIR(sqrt(e / d_params.g11), 0);
	else
		data[index] = MAKE_VALUE_PAIR(0, 0);
}

// Propagates state vector in k-space for steady state calculation
__global__ void propagateKSpace(value_pair *data)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	value_type propG = exp(-d_params.dtGP / 2 * tex1Dfetch(k_tex, index));
	value_pair temp = data[index];
	temp.x *= propG;
	temp.y *= propG;
	data[index] = temp;
}

// Propagates state vector in x-space for steady state calculation
__global__ void propagateToEndpoint(value_pair *data)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a = data[index];

	//store initial x-space field
	value_pair a1 = a;

	value_type da;

	//iterate to midpoint solution
	for(int iter = 0; iter < d_params.itmax; iter++)
	{
		//calculate midpoint log derivative and exponentiate
		da = exp(d_params.dtGP / 2 * (- potential(index) / 2 -
			d_params.g11 * (a.x * a.x + a.y * a.y)));

		//propagate to midpoint using log derivative
		a.x = a1.x * da;
		a.y = a1.y * da;
	}

	//propagate to endpoint using log derivative
	a.x *= da;
	a.y *= da;

	data[index] = a;
}

__global__ void calculateGPEnergy(value_type *res, value_pair *a)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a0 = a[index];
	value_type module = (a0.x * a0.x + a0.y * a0.y);

	//res[index] = module * (d_params.mu - potential(index) / 2 +
	//		d_params.g11 * module / 2);

	res[index] = module * (potential(index) / 2 +
			d_params.g11 * module / 2);
}

__global__ void calculateGPEnergy2(value_pair *a)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a0 = a[index];
	value_type coeff = tex1Dfetch(k_tex, index);
	a0.x *= coeff;
	a0.y *= coeff;
	a[index] = a0;
}

__global__ void calculateGPEnergy3(value_type *res, value_pair *a, value_pair *a_modified)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a0 = a[index];
	value_pair am = a_modified[index];
	res[index] += a0.x * am.x + a0.y * am.y;
}

// Supplementary function for inverse FFT (both cufft and batchfft)
// They are normalized so that ifft(fft(x)) = size(x) * x, and we need
// ifft(fft(x)) = x
__global__ void normalizeInverseFFT(value_pair *data, value_type coeff)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	value_pair temp = data[index];
	temp.x *= coeff;
	temp.y *= coeff;
	data[index] = temp;
}

// Calculate squared modules for two-dimensional vectors
__global__ void calculateModules(value_type *output, value_pair *input)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	value_pair temp = input[index];
	output[index] = temp.x * temp.x + temp.y * temp.y;
}

// Initialize ensembles with steady state + noise for evolution calculation
__global__ void initializeEnsembles(value_pair *a, value_pair *b, value_pair *steady_state, value_pair *noise)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	int single_size = d_params.cells; // number of cells in ensemble
	int size = single_size * d_params.ne; // total number of cells

	value_pair temp;
	value_pair noise_a = noise[index];
	value_pair noise_b = noise[index + size];
	value_pair steady_val = steady_state[index % single_size];

	//Initialises a-ensemble amplitudes with vacuum noise
	temp.x =
		steady_val.x +
		d_params.Va * noise_a.x;
	temp.y =
		steady_val.y +
		d_params.Va * noise_a.y;
	a[index] = temp;

	//Initialises b-ensemble amplitudes with vacuum noise
	temp.x = d_params.Vb * noise_b.x;
	temp.y = d_params.Vb * noise_b.y;
	b[index] = temp;
}

//Apply pi/2 Bragg pulse
__global__ void applyBraggPulse(value_pair *a, value_pair *b)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a0 = a[index];
	value_pair b0 = b[index];

//	a[index] = MAKE_VALUE_PAIR((a0.x + b0.y) / sqrt(2.0), (a0.y - b0.x) / sqrt(2.0));
//	b[index] = MAKE_VALUE_PAIR((b0.x + a0.y) / sqrt(2.0), (b0.y - a0.x) / sqrt(2.0));
	a[index] = cmul(cadd(a0, cmul(b0, MAKE_VALUE_PAIR(0, -1))), 1.0 / sqrt(2.0));
	b[index] = cmul(cadd(cmul(a0, MAKE_VALUE_PAIR(0, -1)), b0), 1.0 / sqrt(2.0));

}

// Linear propagate in k-space for evolution calculation
__global__ void linearPropagate(value_pair *a, value_pair *b, value_type dt)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	int total_pow = d_params.nvx_pow + d_params.nvy_pow;
	value_type propWangle = tex1Dfetch(k_tex, index - ((index >> total_pow) << total_pow)) * dt / 2;

	value_type propWcos = cos(propWangle);
	value_type propWsin = sin(propWangle);

	value_pair a0 = a[index];
	value_pair b0 = b[index];

	a[index] = MAKE_VALUE_PAIR(a0.x * propWcos - a0.y * propWsin,
		a0.y * propWcos + a0.x * propWsin);
	b[index] = MAKE_VALUE_PAIR(b0.x * propWcos - b0.y * propWsin,
		b0.y * propWcos + b0.x * propWsin);
}

// Midpoint propagation for evolution calculation
__global__ void propagateMidpoint(value_pair *a, value_pair *b, value_type dt)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	int total_pow = d_params.nvx_pow + d_params.nvy_pow + d_params.nvz_pow;
	value_type E_middle = - potential(index - ((index >> total_pow) << total_pow)) / 2;
	value_type pa = E_middle + d_params.E / 2;
	value_type pb = E_middle - d_params.E / 2;

	value_pair a0 = a[index];
	value_pair b0 = b[index];

	value_type a0_squared = a0.x * a0.x + a0.y * a0.y;
	value_type b0_squared = b0.x * b0.x + b0.y * b0.y;

	//calculate midpoint log derivative

	value_type da = -d_params.g11 * a0_squared - d_params.g12 * b0_squared + pa;
	value_type db = -d_params.g22 * b0_squared - d_params.g12 * a0_squared + pb;

	value_type a_angle = dt * da;
	value_type cos_a = cos(a_angle);
	value_type sin_a = sin(a_angle);

	value_type b_angle = dt * db;
	value_type cos_b = cos(b_angle);
	value_type sin_b = sin(b_angle);

	//Use log propagator to calculate next time point
	a[index] = MAKE_VALUE_PAIR(
		a0.x * cos_a + a0.y * sin_a,
		a0.y * cos_a - a0.x * sin_a
	);

	b[index] = MAKE_VALUE_PAIR(
		b0.x * cos_b + b0.y * sin_b,
		b0.y * cos_b - b0.x * sin_b
	);
}

// normalize particle density
__global__ void normalizeParticles(value_type *sum_a, value_type *sum_b)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	sum_a[index] = sum_a[index] * d_params.dx * d_params.dy * d_params.dz - d_params.V / 2.0f;
	sum_b[index] = sum_b[index] * d_params.dx * d_params.dy * d_params.dz - d_params.V / 2.0f;
}

__global__ void normalizeParticles2(value_type *sum_a, value_type *sum_b)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_type n1 = sum_a[index] * d_params.dx * d_params.dy * d_params.dz;
	value_type n2 = sum_b[index] * d_params.dx * d_params.dy * d_params.dz;
	sum_a[index] = n1 * n1;
	sum_b[index] = n2 * n2;
}

// Reduces a small set of arrays (if reduce power is too high it is better
// to use transpose() + reduce())
__global__ void smallReduce(value_type *sum, value_type *data, int n)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_type temp = 0;
	int size = blockDim.x * gridDim.x * gridDim.y;
	int current_index = index;

	for(int i = 0; i < n; i++)
	{
		temp += data[current_index];
		current_index += size;
	}

	sum[index] = temp;
}

__device__ __inline__ value_type density(value_pair a)
{
	 return (a.x * a.x + a.y * a.y) * d_params.dx * d_params.dy * d_params.dz - d_params.V / 2.0f;
}

__global__ void calculatePulse(value_type *a_dens, value_type *b_dens, value_pair *a, value_pair *b)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a_val = a[index];
	value_pair b_val = b[index];

//	value_pair a_new = MAKE_VALUE_PAIR((a_val.x + b_val.x) / sqrt(2.0f), (a_val.y + b_val.y) / sqrt(2.0f));
//	value_pair b_new = MAKE_VALUE_PAIR((a_val.x - b_val.x) / sqrt(2.0f), (a_val.y - b_val.y) / sqrt(2.0f));
	value_pair a_new = MAKE_VALUE_PAIR((a_val.x + b_val.y) / sqrt(2.0f), (a_val.y - b_val.x) / sqrt(2.0f));
	value_pair b_new = MAKE_VALUE_PAIR((a_val.y + b_val.x) / sqrt(2.0f), (b_val.y - a_val.x) / sqrt(2.0f));

	a_dens[index] = density(a_new);
	b_dens[index] = density(b_new);
}

// Pi/2 rotate around vector in equatorial plane, with angle alpha between it and x axis
__global__ void halfPiRotate(value_type *a_res, value_type *b_res, value_pair *a, value_pair *b, value_type alpha)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a0 = a[index];
	value_pair b0 = b[index];

	value_type root2 = sqrt(2.0);
	value_type cosa = cos(alpha);
	value_type sina = sin(alpha);

//	a0 = MAKE_VALUE_PAIR((a0.x + b0.y) / root2, (a0.y - b0.x) / root2);
//	b0 = MAKE_VALUE_PAIR((b0.x + a0.y) / root2, (b0.y - a0.x) / root2);
/*
	value_pair a_new = cmul(
		cadd(
			a0,
			cmul(
				b0,
				MAKE_VALUE_PAIR((root2 - 1) * cosa, (1 - root2) * sina)
			)
		),
		1.0 / root2
	);
	value_pair b_new = cmul(
		cadd(
			cmul(
				a0,
				MAKE_VALUE_PAIR(-sina, cosa)
			),
			cmul(
				b0,
				MAKE_VALUE_PAIR(root2, -1)
			)
		),
		1.0 / root2
	);
 */

	value_pair a_new = cmul(cadd(a0, cmul(b0, MAKE_VALUE_PAIR(sina, -cosa))), 1.0 / root2);
	value_pair b_new = cmul(cadd(cmul(a0, MAKE_VALUE_PAIR(-sina, -cosa)), b0), 1.0 / root2);

	a_res[index] = density(a_new);
	b_res[index] = density(b_new);
}

__global__ void calculatePhaseDiff(value_type *res, value_pair *a, value_pair *b)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a_val = a[index];
	value_pair b_val = b[index];

	value_type phase_a = atan2f(a_val.y, a_val.x);
	value_type phase_b = atan2f(b_val.y, b_val.x);

	value_type diff = (phase_a - phase_b);
	if(diff < -M_PI)
		diff += 2 * M_PI;
	else if(diff > M_PI)
		diff -= 2 * M_PI;
	diff = abs(diff);

	diff *= density(a_val) * density(b_val);

	res[index] = diff;
}


// Fill constant memory
void initConstants(CalculationParameters &params)
{
	// copy calculation constants to constant memory
	cutilSafeCall(cudaMemcpyToSymbol(d_params, &params, sizeof(params)));
}
