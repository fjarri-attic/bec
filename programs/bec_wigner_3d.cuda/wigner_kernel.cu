#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include <math.h>

#include "defines.h"

texture<value_type, 1> k_tex;
value_type *k_buf;

// calculation parameters in constant memory, for easy access from kernels
__constant__ CalculationParameters d_params;

// auxiliary function, returning value of wave vector for given lattice index, step and lattice size
value_type getK(int i, value_type dk, int N)
{
	if(2 * i > N)
		return dk * (i - N);
	else
		return dk * i;
}

// initialize texture with wave vector lengths for corresponding spatial lattice nodes
// (i.e., k's which will be used for corresponding lattice nodes after Fourier transform)
void initWaveVectors(CalculationParameters &params)
{
	value_type *h_k = new value_type[params.cells];

	for(int index = 0; index < params.cells; index++)
	{
		int k = index >> (params.nvx_pow + params.nvy_pow);
		int k_shift = (k << (params.nvx_pow + params.nvy_pow));
		int j = (index - k_shift) >> params.nvx_pow;
		int i = (index - k_shift) - (j << params.nvx_pow);

		value_type kx = getK(i, params.dkx, params.nvx);
		value_type ky = getK(j, params.dky, params.nvy);
		value_type kz = getK(k, params.dkz, params.nvz);

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

void releaseWaveVectors()
{
	cudaUnbindTexture(k_tex);
	cudaFree(k_buf);
}

// returns wave vector length from table
__device__ __inline__ value_type getWaveVectorLength(int index)
{
	return tex1Dfetch(k_tex, index);
}


// auxiliary functions for operations with complex values

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

__device__ __inline__ value_type module(value_pair a)
{
	return a.x * a.x + a.y * a.y;
}

// Returns potential energy for given lattice node
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

// fill given buffer with ground state, obtained from Thomas-Fermi approximation
__global__ void fillWithTFGroundState(value_pair *data)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_type e = d_params.mu - potential(index) / 2;
	if(e > 0)
		data[index] = MAKE_VALUE_PAIR(sqrt(e / d_params.g11), 0);
	else
		data[index] = MAKE_VALUE_PAIR(0, 0);
}

// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
__global__ void propagateKSpaceImaginaryTime(value_pair *data)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	value_type prop_coeff = exp(-d_params.dtGP / 2 * getWaveVectorLength(index));
	value_pair temp = data[index];
	data[index] = cmul(temp, prop_coeff);
}

// Propagates state vector in k-space for evolution calculation (i.e., in real time)
__global__ void propagateKSpaceRealTime(value_pair *a, value_pair *b, value_type dt)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	int total_pow = d_params.nvx_pow + d_params.nvy_pow + d_params.nvz_pow;
	int index_in_ensemble = index - ((index >> total_pow) << total_pow);

	value_type prop_angle = getWaveVectorLength(index_in_ensemble) * dt / 2;
	value_pair prop_coeff = MAKE_VALUE_PAIR(cos(prop_angle), sin(prop_angle));

	value_pair a0 = a[index];
	value_pair b0 = b[index];

	a[index] = cmul(a0, prop_coeff);
	b[index] = cmul(b0, prop_coeff);
}

// Propagates state in x-space for steady state calculation
__global__ void propagateXSpaceOneComponent(value_pair *data)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a = data[index];

	//store initial x-space field
	value_pair a0 = a;

	value_type da;
	value_type V = -potential(index) / 2;

	//iterate to midpoint solution
	for(int iter = 0; iter < d_params.itmax; iter++)
	{
		//calculate midpoint log derivative and exponentiate
		da = exp(d_params.dtGP / 2 * (V - d_params.g11 * module(a)));

		//propagate to midpoint using log derivative
		a = cmul(a0, da);
	}

	//propagate to endpoint using log derivative
	a.x *= da;
	a.y *= da;

	data[index] = a;
}

// Propagates state vector in x-space for evolution calculation
__global__ void propagateXSpaceTwoComponent(value_pair *aa, value_pair *bb, value_type dt)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	int total_pow = d_params.nvx_pow + d_params.nvy_pow + d_params.nvz_pow;
	value_type V = -potential(index - ((index >> total_pow) << total_pow)) / 2;

	value_pair a = aa[index];
	value_pair b = bb[index];

	//store initial x-space field
	value_pair a0 = a;
	value_pair b0 = b;

	value_pair da = MAKE_VALUE_PAIR(0, 0), db = MAKE_VALUE_PAIR(0, 0);

	//iterate to midpoint solution
	for(int iter = 0; iter < d_params.itmax; iter++)
	{
		value_type n_a = module(a);
		value_type n_b = module(b);

		value_type pa = V - d_params.g11 * n_a - d_params.g12 * n_b;
		value_type pb = V - d_params.g22 * n_b - d_params.g12 * n_a;

		//calculate midpoint log derivative and exponentiate
		value_type a_angle = dt * pa / 2;
		da = MAKE_VALUE_PAIR(cos(a_angle), -sin(a_angle));

		value_type b_angle = dt * pb / 2;
		db = MAKE_VALUE_PAIR(cos(b_angle), -sin(b_angle));

		//propagate to midpoint using log derivative
		a = cmul(a0, da);
		b = cmul(b0, db);
	}

	//propagate to endpoint using log derivative
	aa[index] = cmul(a, da);
	bb[index] = cmul(b, db);
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

__global__ void calculateChemPotential(value_type *res, value_pair *a)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a0 = a[index];
	value_type module = (a0.x * a0.x + a0.y * a0.y);

	//res[index] = module * (d_params.mu - potential(index) / 2 +
	//		d_params.g11 * module / 2);

	res[index] = module * (potential(index) / 2 +
			d_params.g11 * module);
}

__global__ void calculateGPEnergy2(value_pair *a)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a0 = a[index];
	value_type coeff = getWaveVectorLength(index);
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
