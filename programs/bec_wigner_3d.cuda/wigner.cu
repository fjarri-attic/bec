#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include <math.h>

#include "defines.h"
#include "batchfft.h"
#include "reduce.cuh"
#include "spectre.h"
#include "transpose.cuh"
#include "wigner_kernel.cu"
#include "cudatexture.h"

// fill complex vector with normally distributed random numbers
void fillWithNormalDistribution(CudaBuffer<value_pair> &data, value_type dev)
{
	value_pair *h_data = new value_pair[data.len()];

	for(int i = 0; i < data.len(); i++)
	{
		// pair of independent uniformly distributed random numbers
		value_pair num;
		num.x = ((value_type)(rand() + 1))/((unsigned int)RAND_MAX + 1);
		num.y = ((value_type)(rand() + 1))/((unsigned int)RAND_MAX + 1);

		// Box-Muller transform to get two normally distributed random numbers
		h_data[i].x = dev * sqrt(-2.0 * log(num.x)) * cos(2 * M_PI * num.y);
		h_data[i].y = dev * sqrt(-2.0 * log(num.x)) * sin(2 * M_PI * num.y);
	}

	data.copyFrom(h_data);

	delete[] h_data;
}

void calculateSteadyState(value_pair *h_steady_state, CalculationParameters &params)
{
	cufftHandle plan;
	dim3 block, grid;
	value_type E = 0;

	CudaBuffer<value_pair> a(params.cells), a_modified(params.cells);
	CudaBuffer<value_type> a_module(params.cells), temp(params.cells);

	createKernelParams(block, grid, params.cells, MAX_THREADS_NUM);
	cufftSafeCall(cufftPlan3d(&plan, params.nvz, params.nvy, params.nvx, CUFFT_C2C));

	//initial GP solution in k-space
	//initialState<<<grid, block>>>(a);
	//cutilCheckMsg("initialState");
	fillWithTFGroundState<<<grid, block>>>(a);
	cutilCheckMsg("fillWithTFGroundState");

	calculateModules<<<grid, block>>>(a_module, a);
	cutilCheckMsg("calculateModules");
	value_type current_N = reduce<value_type>(a_module, temp, params.cells, 1) *
		params.dx * params.dy * params.dz;
	printf("N = %f\n", current_N);
	normalizeInverseFFT<<<grid, block>>>(a, sqrt(params.N / current_N));

	cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_INVERSE));
	normalizeInverseFFT<<<grid, block>>>(a, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");

	//////////////////////////////////////////////////////////////////////////
	// 	Starts GP loop in time: calculate mean-field steady-state
	//////////////////////////////////////////////////////////////////////////
	value_type t = 0;
	while(1)
	{
		// Linear propagate in k-space
		propagateKSpaceImaginaryTime<<<grid, block>>>(a);
		cutilCheckMsg("propagateKSpaceImaginaryTime");

		// FFT into x-space
		cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_FORWARD));

		propagateToEndpoint<<<grid, block>>>(a);
		cutilCheckMsg("propagateToEndpoint");

		// FFT into k-space
		cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_INVERSE));
		normalizeInverseFFT<<<grid, block>>>(a, 1.0 / params.cells);
		cutilCheckMsg("normalizeInverseFFT");

		// Linear propagate in k-space
		propagateKSpaceImaginaryTime<<<grid, block>>>(a);
		cutilCheckMsg("propagateKSpaceImaginaryTime");


		// FFT into x-space
		cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_FORWARD));

		// Normalize
		calculateModules<<<grid, block>>>(a_module, a);
		cutilCheckMsg("calculateModules");
		current_N = reduce<value_type>(a_module, temp, params.cells, 1) *
			params.dx * params.dy * params.dz;
		printf("N = %f\n", current_N);
		normalizeInverseFFT<<<grid, block>>>(a, sqrt(params.N / current_N));

		// Calculate energy
		calculateGPEnergy<<<grid, block>>>(a_module, a);

		cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a_modified, CUFFT_INVERSE));
		normalizeInverseFFT<<<grid, block>>>(a_modified, 1.0 / params.cells);
		cutilCheckMsg("normalizeInverseFFT");
		calculateGPEnergy2<<<grid, block>>>(a_modified);
		cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a_modified, (cufftComplex*)a_modified, CUFFT_FORWARD));
		calculateGPEnergy3<<<grid, block>>>(a_module, a, a_modified);

		value_type new_E = reduce<value_type>(a_module, temp, params.cells, 1);

		// FFT into k-space
		cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_INVERSE));
		normalizeInverseFFT<<<grid, block>>>(a, 1.0 / params.cells);
		cutilCheckMsg("normalizeInverseFFT");

		if(abs((new_E - E) / new_E) < 0.000001)
			break;

		E = new_E;
		t += params.dtGP;
	} //end time loop

	//FFT into x-space
	cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_FORWARD));

	// DEBUG
	//fillWithTFGroundState<<<grid, block>>>(a);
	//cutilCheckMsg("fillWithTFGroundState");

	// save steady state
	a.copyTo(h_steady_state);

	// Calculate number of atoms in steady state (for debug purposes)
	calculateModules<<<grid, block>>>(a_module, a);
	printf("%f\n", reduce<value_type>(a_module, temp, params.cells, 1) * params.dx * params.dy * params.dz);

	calculateModules<<<grid, block>>>(a_module, a);
	cutilCheckMsg("calculateModules");
	current_N = reduce<value_type>(a_module, temp, params.cells, 1) *
		params.dx * params.dy * params.dz;
	normalizeInverseFFT<<<grid, block>>>(a, sqrt(params.N / current_N));

	calculateGPEnergy<<<grid, block>>>(a_module, a);
	cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a_modified, CUFFT_INVERSE));
	normalizeInverseFFT<<<grid, block>>>(a_modified, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");
	calculateGPEnergy2<<<grid, block>>>(a_modified);
	cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a_modified, (cufftComplex*)a_modified, CUFFT_FORWARD));
	calculateGPEnergy3<<<grid, block>>>(a_module, a, a_modified);

	value_type new_E = reduce<value_type>(a_module, temp, params.cells, 1);
	printf("E = %f\n", new_E / (2 * M_PI * params.fz * params.N) * params.dx * params.dy * params.dz);

	calculateChemPotential<<<grid, block>>>(a_module, a);
	cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a_modified, CUFFT_INVERSE));
	normalizeInverseFFT<<<grid, block>>>(a_modified, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");
	calculateGPEnergy2<<<grid, block>>>(a_modified);
	cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)a_modified, (cufftComplex*)a_modified, CUFFT_FORWARD));
	calculateGPEnergy3<<<grid, block>>>(a_module, a, a_modified);

	value_type new_mu = reduce<value_type>(a_module, temp, params.cells, 1);
	printf("mu = %f\n", new_mu / (2 * M_PI * params.fz * params.N) * params.dx * params.dy * params.dz);

	cufftDestroy(plan);
}


void calculateAverage(CalculationParameters &params, EvolutionState &state)
{
	calculateModules<<<state.grid, state.block>>>(state.dens_b, state.b);
	cutilCheckMsg("calculateModules");
	calculateModules<<<state.grid, state.block>>>(state.dens_a, state.a);
	cutilCheckMsg("calculateModules");
	normalizeParticles<<<state.grid, state.block>>>(state.dens_a, state.dens_b);
	cutilCheckMsg("normalizeParticles");

	value_type sum1 = 0, sum2 = 0;

	for(int i = 0; i < params.ne; i++)
	{
		value_type n1 = reduce<value_type>(state.dens_a + i * params.cells, state.temp, params.cells, 1);
		value_type n2 = reduce<value_type>(state.dens_b + i * params.cells, state.temp, params.cells, 1);
		sum1 += -n1 * n1;
		sum2 += -n2 * n2;
	}

	calculateModules<<<state.grid, state.block>>>(state.dens_b, state.b);
	cutilCheckMsg("calculateModules");
	calculateModules<<<state.grid, state.block>>>(state.dens_a, state.a);
	cutilCheckMsg("calculateModules");
	normalizeParticles2<<<state.grid, state.block>>>(state.dens_a, state.dens_b);
	cutilCheckMsg("normalizeParticles2");

	for(int i = 0; i < params.ne; i++)
	{
		value_type n1 = reduce<value_type>(state.dens_a + i * params.cells, state.temp, params.cells, 1);
		value_type n2 = reduce<value_type>(state.dens_b + i * params.cells, state.temp, params.cells, 1);
		sum1 += n1;
		sum2 += n2;
	}

	printf("%f %f\n", sum1 / params.ne, sum2 / params.ne);
}

// Propagate k-state for evolution calculation
void propagate(CalculationParameters &params, EvolutionState &state, value_type dt)
{
	propagateKSpaceRealTime<<<state.grid, state.block>>>(state.a, state.b, dt);
	cutilCheckMsg("propagateKSpaceRealTime");

	//FFT into x-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_FORWARD));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_FORWARD));
/*
	calculateModules<<<state.grid, state.block>>>(state.dens_b, state.b);
	cutilCheckMsg("calculateModules");
	calculateModules<<<state.grid, state.block>>>(state.dens_a, state.a);
	cutilCheckMsg("calculateModules");
	normalizeParticles<<<state.grid, state.block>>>(state.dens_a, state.dens_b);
	cutilCheckMsg("normalizeParticles");
	printf("%f %f\n",
	       reduce<value_type>(state.dens_a, state.temp, params.cells, 1),
	       reduce<value_type>(state.dens_b, state.temp, params.cells, 1));
*/
//	calculateAverage(params, state);

	propagateMidpoint<<<state.grid, state.block>>>(state.a, state.b, dt);
	cutilCheckMsg("propagateMidpoint");

	//FFT into k-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_INVERSE));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_INVERSE));
	normalizeInverseFFT<<<state.grid, state.block>>>(state.a, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");
	normalizeInverseFFT<<<state.grid, state.block>>>(state.b, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");

	//Linear propagate a,b-field
	propagateKSpaceRealTime<<<state.grid, state.block>>>(state.a, state.b, dt);
	cutilCheckMsg("propagateKSpaceRealTime");
}

// initialize evolution state
void initEvolution(value_pair *h_steady_state, CalculationParameters &params, EvolutionState &state)
{
	int size = params.cells * params.ne;
	CudaBuffer<value_pair> noise(size * 2), steady_state(params.cells);

	steady_state.copyFrom(h_steady_state);

	// initialize ensembles
	srand(time(0));
	fillWithNormalDistribution(noise, 0.5 / sqrt(params.dx * params.dy * params.dz));
	initializeEnsembles<<<state.grid, state.block>>>(state.a, state.b, steady_state, noise);
	cutilCheckMsg("initializeEnsembles");

	// FFT into k-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_INVERSE));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_INVERSE));
	normalizeInverseFFT<<<state.grid, state.block>>>(state.a, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");
	normalizeInverseFFT<<<state.grid, state.block>>>(state.b, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");

	// Equilibration phase
//	for(value_type t = 0; t <= params.tmaxWig; t += params.dtWig)
//		propagate(params, state, params.dtWig);

	applyBraggPulse<<<state.grid, state.block>>>(state.a, state.b);
	cutilCheckMsg("applyBraggPulse");
}

// propagate system and fill current state graph data
void calculateEvolution(CalculationParameters &params, EvolutionState &state, value_type dt)
{
	propagate(params, state, dt);
	state.t += dt;

	//FFT into x-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_FORWARD));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_FORWARD));
	cutilSafeCall(cudaThreadSynchronize());
/*
	halfPiRotate<<<state.grid, state.block>>>(state.dens_a, state.dens_b, state.a, state.b, 0);
	cutilCheckMsg("halfPiRotate");
	value_type na = reduce<value_type>(state.dens_a, state.temp, params.cells * params.ne, 1) / params.ne;
	value_type nb = reduce<value_type>(state.dens_b, state.temp, params.cells * params.ne, 1) / params.ne;
	printf("%f %f\n", state.t, na+nb);
 */
	//calculateModules<<<state.grid, state.block>>>(state.dens_a, state.a);
	//calculateModules<<<state.grid, state.block>>>(state.dens_b, state.b);
	//normalizeParticles<<<state.grid, state.block>>>(state.dens_a, state.dens_b);
	halfPiRotate<<<state.grid, state.block>>>(state.dens_a, state.dens_b, state.a, state.b, 0);
	cutilCheckMsg("halfPiRotate");
/*
	value_type max = 0;
	for(value_type alpha = 0; alpha < 2 * M_PI; alpha += 0.5)
	{
		halfPiRotate<<<state.grid, state.block>>>(state.dens_a, state.dens_b, state.a, state.b, alpha);
		cutilCheckMsg("halfPiRotate");
		value_type N1 = reduce<value_type>(state.dens_a, state.temp, params.cells * params.ne, 1);
		value_type N2 = reduce<value_type>(state.dens_b, state.temp, params.cells * params.ne, 1);
		value_type res = abs(N1 - N2) / (N1 + N2);
		if(res > max)
			max = res;
	}
	printf("%f %f\n", state.t, max);
*/
	//FFT into k-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_INVERSE));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_INVERSE));

	cutilSafeCall(cudaThreadSynchronize());

	normalizeInverseFFT<<<state.grid, state.block>>>(state.a, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");
	normalizeInverseFFT<<<state.grid, state.block>>>(state.b, 1.0 / params.cells);
	cutilCheckMsg("normalizeInverseFFT");

	// reduce<value_type>() reduces neighbouring values first, and we need to reduce
	// values for each particle separately
	// so we transform [ensemble1, ensemble2, ...] to [particle1, particle2, ...] using transpose<value_type>()
	// and then perform reduce<value_type>()

	if(params.ne >= 16)
	{
		cutilSafeCall(transpose<value_type>(state.temp, state.dens_a, params.cells, params.ne, 1));
		reduce<value_type>(state.temp, state.dens_a, params.cells * params.ne, params.cells);
	}
	else
	{
		dim3 block, grid;
		createKernelParams(block, grid, params.cells, MAX_THREADS_NUM);
		smallReduce<<<grid, block>>>(state.temp, state.dens_a, params.ne);
		cutilCheckMsg("smallReduce");
	}

	// for XY slice just copy memory from the middle
	//state.dens_a_xy.copyFrom(state.temp, params.nvx * params.nvy, params.nvx * params.nvy * (params.nvz / 2));

	// projection on XY plane
	cutilSafeCall(transpose<value_type>(state.temp2, state.temp, params.nvx * params.nvy, params.nvz, 1));
	reduce<value_type>(state.temp2, state.dens_a, params.cells, params.nvx * params.nvy);
	state.dens_a_xy.copyFrom(state.temp2, params.nvx * params.nvy);

	// for YZ slice copy memory after transpose
	//cutilSafeCall(transpose<value_type>(state.dens_a, state.temp, params.nvx, params.nvy * params.nvz, 1));
	//state.temp.copyFrom(state.dens_a, params.nvy * params.nvz, params.nvy * params.nvz * (params.nvx / 2));
	//cutilSafeCall(transpose<value_type>(state.dens_a_zy, state.temp, params.nvy, params.nvz, 1));

	// projection on YZ plane
	reduce<value_type>(state.temp, state.temp2, params.cells, params.nvy * params.nvz);
	state.dens_a_zy.copyFrom(state.temp2, params.nvy * params.nvz);
	cutilSafeCall(transpose<value_type>(state.dens_a_zy, state.temp, params.nvy, params.nvz, 1));

	if(params.ne >= 16)
	{
		cutilSafeCall(transpose<value_type>(state.temp, state.dens_b, params.cells, params.ne, 1));
		reduce<value_type>(state.temp, state.dens_b, params.cells * params.ne, params.cells);
	}
	else
	{
		dim3 block, grid;
		createKernelParams(block, grid, params.cells, MAX_THREADS_NUM);
		smallReduce<<<grid, block>>>(state.temp, state.dens_b, params.ne);
		cutilCheckMsg("smallReduce");
	}

	// for XY slice just copy memory from the middle
	//state.dens_b_xy.copyFrom(state.temp, params.nvx * params.nvy, params.nvx * params.nvy * (params.nvz / 2));

	// projection on XY plane
	cutilSafeCall(transpose<value_type>(state.temp2, state.temp, params.nvx * params.nvy, params.nvz, 1));
	reduce<value_type>(state.temp2, state.dens_b, params.cells, params.nvx * params.nvy);
	state.dens_b_xy.copyFrom(state.temp2, params.nvx * params.nvy);

	// for YZ slice copy memory after transpose
	//cutilSafeCall(transpose<value_type>(state.dens_b, state.temp, params.nvx, params.nvy * params.nvz, 1));
	//state.temp.copyFrom(state.dens_b, params.nvy * params.nvz, params.nvy * params.nvz * (params.nvx / 2));
	//cutilSafeCall(transpose<value_type>(state.dens_b_zy, state.temp, params.nvy, params.nvz, 1));

	// projection on YZ plane
	reduce<value_type>(state.temp, state.temp2, params.cells, params.nvy * params.nvz);
	state.dens_b_zy.copyFrom(state.temp2, params.nvy * params.nvz);
	cutilSafeCall(transpose<value_type>(state.dens_b_zy, state.temp, params.nvy, params.nvz, 1));
}

// Draw graphs from current state to provided buffers
void drawState(CalculationParameters &params, EvolutionState &state, CudaTexture &a_xy_tex,
	CudaTexture &b_xy_tex, CudaTexture &a_zy_tex, CudaTexture &b_zy_tex)
{
	value_type scale = 5.0 * params.N * params.ne / params.cells;
	value_type xy_scale = scale * params.nvz;
	value_type zy_scale = scale * params.nvx;

	drawData(a_xy_tex, state.dens_a_xy, xy_scale);
	drawData(b_xy_tex, state.dens_b_xy, xy_scale);
	drawData(a_zy_tex, state.dens_a_zy, zy_scale);
	drawData(b_zy_tex, state.dens_b_zy, zy_scale);

	float4 *b_xy_buf = b_xy_tex.map();
	cudaMemcpy(state.to_bmp, b_xy_buf, params.nvx * params.nvy * sizeof(float4), cudaMemcpyDeviceToHost);
	b_xy_tex.unmap();
}
