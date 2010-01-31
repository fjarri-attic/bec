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

		h_k[index] = (kx * kx + ky * ky + kz * kz) / 2;
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

__device__ __inline__ value_pair cexp(value_pair a)
{
	value_type module = exp(a.x);
	return MAKE_VALUE_PAIR(module * cos(a.y), module * sin(a.y));
}

// Returns external potential energy for given lattice node
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

	return (x * x + y * y + z * z / (d_params.lambda * d_params.lambda)) / 2;
}

// fill given buffer with ground state, obtained from Thomas-Fermi approximation
__global__ void fillWithTFGroundState(value_pair *data)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_type e = d_params.mu - potential(index);
	if(e > 0)
		data[index] = MAKE_VALUE_PAIR(sqrt(e / d_params.g11), 0);
	else
		data[index] = MAKE_VALUE_PAIR(0, 0);
}

// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
__global__ void propagateKSpaceImaginaryTime(value_pair *data)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	value_type prop_coeff = exp(-d_params.dt_steady / 2 * getWaveVectorLength(index));
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
	value_type V = potential(index);

	//iterate to midpoint solution
	for(int iter = 0; iter < d_params.itmax; iter++)
	{
		//calculate midpoint log derivative and exponentiate
		da = exp(d_params.dt_steady / 2 * (-V - d_params.g11 * module(a)));

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
	value_type V = potential(index - ((index >> total_pow) << total_pow));

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

		// TODO: there must be no minus sign before imaginary part,
		// but without it the whole thing diverges
		value_pair pa = MAKE_VALUE_PAIR(
			-(d_params.l111 * n_a * n_a + d_params.l12 * n_b) / 2,
			//0,
			-(-V - d_params.g11 * n_a - d_params.g12 * n_b));
		value_pair pb = MAKE_VALUE_PAIR(
			-(d_params.l22 * n_b + d_params.l12 * n_a) / 2,
			//0,
			-(-V - d_params.g22 * n_b - d_params.g12 * n_a - d_params.detuning));

		// calculate midpoint log derivative and exponentiate
		da = cexp(cmul(pa, dt / 2));
		db = cexp(cmul(pb, dt / 2));

		//propagate to midpoint using log derivative
		a = cmul(a0, da);
		b = cmul(b0, db);
	}

	//propagate to endpoint using log derivative
	aa[index] = cmul(a, da);
	bb[index] = cmul(b, db);
}

// auxiliary function for calculating nonlinear component of state energy integral
__global__ void calculateNonlinearEnergyPart(value_type *res, value_pair *a)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_type n_a = module(a[index]);
	res[index] = n_a * (potential(index) + d_params.g11 * n_a / 2);
}

// auxiliary function for calculating nonlinear component of chemical potential integral
__global__ void calculateNonlinearMuPart(value_type *res, value_pair *a)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_type n_a = module(a[index]);
	res[index] = n_a * (potential(index) + d_params.g11 * n_a);
}

// auxiliary function for calculating energy/chem.potential integral
__global__ void combineNonlinearAndDifferential(value_type *res, value_pair *state, value_pair *fourier_state)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	value_pair differential = cmul(fourier_state[index], getWaveVectorLength(index));
	value_pair temp = cmul(state[index], differential);

	// temp.y will be equal to 0, because \psi * D \psi is a real number
	res[index] += temp.x;
}

// componentwise multiplication
__global__ void multiply(value_pair *data, value_type coeff)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	data[index] = cmul(data[index], coeff);
}

// componentwise multiplication, for two data sets
// (in order to reduce number of calls to GPU)
__global__ void multiplyPair(value_pair *data1, value_pair *data2, value_type coeff)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	data1[index] = cmul(data1[index], coeff);
	data2[index] = cmul(data2[index], coeff);
}

// Calculate squared modules for two-dimensional vectors
__global__ void calculateModules(value_type *output, value_pair *input)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	output[index] = module(input[index]);
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
	temp.x = steady_val.x +	d_params.Va * noise_a.x;
	temp.y = steady_val.y +	d_params.Va * noise_a.y;
	a[index] = temp;

	//Initialises b-ensemble amplitudes with vacuum noise
	temp.x = d_params.Vb * noise_b.x;
	temp.y = d_params.Vb * noise_b.y;
	b[index] = temp;
}

// Apply pi/2 pulse (instantaneus approximation)
__global__ void applyHalfPiPulse(value_pair *a, value_pair *b)
{
	int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

	value_pair a0 = a[index];
	value_pair b0 = b[index];

	a[index] = cmul(cadd(a0, cmul(b0, MAKE_VALUE_PAIR(0, -1))), 1.0 / sqrt(2.0));
	b[index] = cmul(cadd(cmul(a0, MAKE_VALUE_PAIR(0, -1)), b0), 1.0 / sqrt(2.0));
}

__device__ __inline__ value_type density(value_pair a)
{
	 return module(a) - d_params.V / 2.0f;
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

	value_pair a_new = cmul(cadd(a0, cmul(b0, MAKE_VALUE_PAIR(sina, -cosa))), 1.0 / root2);
	value_pair b_new = cmul(cadd(cmul(a0, MAKE_VALUE_PAIR(-sina, -cosa)), b0), 1.0 / root2);

	a_res[index] = density(a_new);
	b_res[index] = density(b_new);
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

// Fill constant memory
void initConstants(CalculationParameters &params)
{
	// copy calculation constants to constant memory
	cutilSafeCall(cudaMemcpyToSymbol(d_params, &params, sizeof(params)));
}
