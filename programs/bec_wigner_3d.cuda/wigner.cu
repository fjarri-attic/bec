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

// returns value of energy/chem.potential integral over given state
// *_temp values are spoiled after call
value_type calculateStateIntegral(CudaBuffer<value_pair> &state,
	CudaBuffer<value_pair> &complex_temp,
	CudaBuffer<value_type> &real_temp,
	CudaBuffer<value_type> &real_temp2,
	CalculationParameters &params,
	batchfftHandle plan, bool energy)
{
	dim3 grid, block;
	createKernelParams(block, grid, state.len(), MAX_THREADS_NUM);

	cufftSafeCall(batchfftExecute(plan, (cufftComplex*)state, (cufftComplex*)complex_temp, CUFFT_INVERSE));
	multiply<<<grid, block>>>(complex_temp, 1.0 / state.len());
	cutilCheckMsg("multiply");

	if(energy)
	{
		calculateNonlinearEnergyPart<<<grid, block>>>(real_temp, state);
		cutilCheckMsg("calculateNonlinearEnergyPart");
	}
	else
	{
		calculateNonlinearMuPart<<<grid, block>>>(real_temp, state);
		cutilCheckMsg("calculateNonlinearMuPart");
	}

	combineNonlinearAndDifferential<<<grid, block>>>(real_temp, state, complex_temp);
	cutilCheckMsg("combineNonlinearAndDifferential");

	return reduce<value_type>(real_temp, real_temp2, state.len(), 1) * params.dx * params.dy * params.dz;
}

// returns number if particles for given state
// *_temp values are spoiled after call
value_type calculateParticles(CudaBuffer<value_pair> &state, CudaBuffer<value_type> &real_temp,
	CudaBuffer<value_type> &real_temp2, CalculationParameters &params)
{
	dim3 grid, block;
	createKernelParams(block, grid, state.len(), MAX_THREADS_NUM);

	calculateModules<<<grid, block>>>(real_temp, state);
	cutilCheckMsg("calculateModules");

	return reduce<value_type>(real_temp, real_temp2, state.len(), 1) *
		params.dx * params.dy * params.dz;
}

void calculateSteadyState(value_pair *h_steady_state, CalculationParameters &params)
{
	batchfftHandle plan;
	dim3 block, grid;
	value_type E = 0;

	CudaBuffer<value_pair> a(params.cells), complex_temp(params.cells);
	CudaBuffer<value_type> real_temp(params.cells), real_temp2(params.cells);

	createKernelParams(block, grid, params.cells, MAX_THREADS_NUM);
	cufftSafeCall(batchfftPlan3d(&plan, params.nvz, params.nvy, params.nvx, CUFFT_C2C, 1));

	//initial GP solution in k-space
	fillWithTFGroundState<<<grid, block>>>(a);
	cutilCheckMsg("fillWithTFGroundState");

	// normalize initial conditions
	value_type N = calculateParticles(a, real_temp, real_temp2, params);
	printf("N = %f\n", N);
	multiply<<<grid, block>>>(a, sqrt(params.N / N));

	// FFT into k-space
	cufftSafeCall(batchfftExecute(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_INVERSE));
	multiply<<<grid, block>>>(a, 1.0 / params.cells);
	cutilCheckMsg("multiply");

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
		cufftSafeCall(batchfftExecute(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_FORWARD));

		propagateXSpaceOneComponent<<<grid, block>>>(a);
		cutilCheckMsg("propagateToEndpoint");

		// FFT into k-space
		cufftSafeCall(batchfftExecute(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_INVERSE));
		multiply<<<grid, block>>>(a, 1.0 / params.cells);
		cutilCheckMsg("multiply");

		// Linear propagate in k-space
		propagateKSpaceImaginaryTime<<<grid, block>>>(a);
		cutilCheckMsg("propagateKSpaceImaginaryTime");

		// Renormalization

		// FFT into x-space
		cufftSafeCall(batchfftExecute(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_FORWARD));

		// Normalize
		N = calculateParticles(a, real_temp, real_temp2, params);
		//printf("N = %f\n", N);
		multiply<<<grid, block>>>(a, sqrt(params.N / N));

		// Calculate energy
		value_type new_E = calculateStateIntegral(a, complex_temp, real_temp, real_temp2, params, plan, true);

		// FFT into k-space
		cufftSafeCall(batchfftExecute(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_INVERSE));
		multiply<<<grid, block>>>(a, 1.0 / params.cells);
		cutilCheckMsg("multiply");

		if(abs((new_E - E) / new_E) < 0.000001)
			break;

		E = new_E;
		t += params.dt_steady;
	} //end time loop

	//FFT into x-space
	cufftSafeCall(batchfftExecute(plan, (cufftComplex*)a, (cufftComplex*)a, CUFFT_FORWARD));

	// save steady state
	a.copyTo(h_steady_state);

	E = calculateStateIntegral(a, complex_temp, real_temp, real_temp2, params, plan, true);
	printf("E = %f\n", E / params.N);

	value_type mu = calculateStateIntegral(a, complex_temp, real_temp, real_temp2, params, plan, false);
	printf("mu = %f\n", mu / params.N);

	batchfftDestroy(plan);
}

// Propagate k-state for evolution calculation
void propagate(CalculationParameters &params, EvolutionState &state, value_type dt)
{
	propagateKSpaceRealTime<<<state.grid, state.block>>>(state.a, state.b, dt);
	cutilCheckMsg("propagateKSpaceRealTime");

	//FFT into x-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_FORWARD));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_FORWARD));

	propagateXSpaceTwoComponent<<<state.grid, state.block>>>(state.a, state.b, dt);
	cutilCheckMsg("propagateMidpoint");

	//FFT into k-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_INVERSE));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_INVERSE));
	multiplyPair<<<state.grid, state.block>>>(state.a, state.b, 1.0 / params.cells);
	cutilCheckMsg("multiplyPair");

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
	fillWithNormalDistribution(noise, 0.5f);
	initializeEnsembles<<<state.grid, state.block>>>(state.a, state.b, steady_state, noise);
	cutilCheckMsg("initializeEnsembles");

	// FFT into k-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_INVERSE));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_INVERSE));
	multiplyPair<<<state.grid, state.block>>>(state.a, state.b, 1.0 / params.cells);
	cutilCheckMsg("multiplyPair");

	// Equilibration phase
	for(value_type t = 0; t <= params.t_equilib; t += params.dt_evo)
		propagate(params, state, params.dt_evo);

	applyHalfPiPulse<<<state.grid, state.block>>>(state.a, state.b);
	cutilCheckMsg("applyBraggPulse");
}

// reduce sparse elements instead of neighbouring ones
// data in *in is spoiled after call
template<class T>
void sparseReduce(CudaBuffer<T> &out, CudaBuffer<T> &in, int length, int final_length = 1)
{
	int coeff = length / final_length;

	if(coeff == 1)
	{
		out.copyFrom(in);
		return;
	}

	// transpose cannot handle matrices with dimensions less than 16
	if(coeff >= 16)
	{
		cutilSafeCall(transpose<T>(out, in, final_length, coeff, 1));
		reduce<T>(out, in, length, final_length);
	}
	else
	{
		dim3 block, grid;
		createKernelParams(block, grid, final_length, MAX_THREADS_NUM);
		smallReduce<T><<<grid, block>>>(out, in, coeff);
		cutilCheckMsg("smallReduce");
	}
}

value_type getComponentRatio(CalculationParameters &params, EvolutionState &state, value_type angle)
{
	halfPiRotate<<<state.grid, state.block>>>(state.dens_a, state.dens_b, state.a, state.b, angle);
	cutilCheckMsg("halfPiRotate");

	value_type N1 = reduce<value_type>(state.dens_a, state.temp, params.cells * params.ne, 1);
	value_type N2 = reduce<value_type>(state.dens_b, state.temp, params.cells * params.ne, 1);
	return (N1 - N2) / (N1 + N2);
}

value_type getVisibility(CalculationParameters &params, EvolutionState &state)
{
	value_type max = 0;
	for(value_type alpha = 0; alpha < 2 * M_PI; alpha += 0.5)
	{
		value_type ratio = getComponentRatio(params, state, alpha);
		if(abs(ratio) > max)
			max = abs(ratio);
	}
	return max;
}

void printComponentRatioAxialProjection(CalculationParameters &params, EvolutionState &state)
{
	halfPiRotate<<<state.grid, state.block>>>(state.dens_a, state.dens_b, state.a, state.b, 0);
	cutilCheckMsg("halfPiRotate");

	sparseReduce(state.temp, state.dens_a, params.cells * params.ne, params.cells);
	reduce<value_type>(state.temp, state.dens_a, params.cells, params.nvz);

	sparseReduce(state.temp2, state.dens_b, params.cells * params.ne, params.cells);
	reduce<value_type>(state.temp2, state.dens_b, params.cells, params.nvz);

	value_type *a_proj = new value_type[params.nvz];
	value_type *b_proj = new value_type[params.nvz];

	state.temp.copyTo(a_proj, params.nvz);
	state.temp2.copyTo(b_proj, params.nvz);

	printf("%f", state.t * params.t_rho * 1000);
	for(int i = 0; i < params.nvz; i++)
		printf(" %f", (a_proj[i] - b_proj[i]) / (a_proj[i] + b_proj[i]));
	printf("\n");

	delete[] a_proj;
	delete[] b_proj;
}

void calculateAverages(CalculationParameters &params, EvolutionState &state)
{
	// calculate norm (sum (|psi|^2) = N)
	calculateModules<<<state.grid, state.block>>>(state.dens_a, state.a);
	cutilCheckMsg("calculateModules");
	value_type a_avg_module = reduce<value_type>(state.dens_a, state.temp, params.cells * params.ne, 1) /
		(params.cells * params.ne);

	calculateModules<<<state.grid, state.block>>>(state.dens_b, state.b);
	cutilCheckMsg("calculateModules");
	value_type b_avg_module = reduce<value_type>(state.dens_b, state.temp, params.cells * params.ne, 1) /
		(params.cells * params.ne);

	printf("%f\n", (a_avg_module + b_avg_module) * params.cells * params.dx * params.dy * params.dz);
	value_type norm = params.N / ((a_avg_module + b_avg_module) * params.cells);

	// Calculate averages
	cutilSafeCall(cudaMemcpy(state.complex_t1, state.a, params.cells * params.ne * sizeof(value_pair), cudaMemcpyDeviceToDevice));
	sparseReduce<value_pair>(state.complex_t2, state.complex_t1, params.cells * params.ne, params.cells);
	kernelAverage<<<state.grid, state.block>>>(state.complex_t1, state.complex_t2, state.a);
	value_pair a_avg = reduce<value_pair>(state.complex_t1, state.complex_t2, params.cells * params.ne, 1) /
		(params.cells * params.ne);

	cutilSafeCall(cudaMemcpy(state.complex_t1, state.b, params.cells * params.ne * sizeof(value_pair), cudaMemcpyDeviceToDevice));
	sparseReduce<value_pair>(state.complex_t2, state.complex_t1, params.cells * params.ne, params.cells);
	kernelAverage<<<state.grid, state.block>>>(state.complex_t1, state.complex_t2, state.b);
	value_pair b_avg = reduce<value_pair>(state.complex_t1, state.complex_t2, params.cells * params.ne, 1) /
		(params.cells * params.ne);

	printf("Avgs: Re(a): %f, Im(a): %f, Re(b): %f, Im(b): %f\n",
	       a_avg.x * norm, a_avg.y * norm, b_avg.x * norm, b_avg.y * norm);
//	printf("Avgs: Re = %f, Im = %f\n", (a_re_avg + b_re_avg) * norm, (a_im_avg + b_im_avg) * norm);
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
	if(!state.pi_pulse_applied && state.t * params.t_rho >= 0.03)
	{
		state.pi_pulse_applied = true;
		applyPiPulse<<<state.grid, state.block>>>(state.a, state.b);
		cutilCheckMsg("applyPiPulse");
	}
 */
//	printf("%f %f\n", state.t * params.t_rho * 1000, getVisibility(params, state));
//	printf("%f %f\n", state.t * params.t_rho * 1000, getComponentRatio(params, state, 0));
	printComponentRatioAxialProjection(params, state);

	// second pi/2 pulse
	halfPiRotate<<<state.grid, state.block>>>(state.dens_a, state.dens_b, state.a, state.b, 0);
	cutilCheckMsg("halfPiRotate");

	//FFT into k-space
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.a, (cufftComplex*)state.a, CUFFT_INVERSE));
	cufftSafeCall(batchfftExecute(state.plan, (cufftComplex*)state.b, (cufftComplex*)state.b, CUFFT_INVERSE));

	cutilSafeCall(cudaThreadSynchronize());

	multiplyPair<<<state.grid, state.block>>>(state.a, state.b, 1.0 / params.cells);
	cutilCheckMsg("multiplyPair");

	// reduce<value_type>() reduces neighbouring values first, and we need to reduce
	// values for each particle separately
	// so we transform [ensemble1, ensemble2, ...] to [particle1, particle2, ...] using transpose<value_type>()
	// and then perform reduce<value_type>()
	sparseReduce(state.temp, state.dens_a, params.cells * params.ne, params.cells);

	// projection on XY plane
	cutilSafeCall(transpose<value_type>(state.temp2, state.temp, params.nvx * params.nvy, params.nvz, 1));
	reduce<value_type>(state.temp2, state.dens_a, params.cells, params.nvx * params.nvy);
	state.dens_a_xy.copyFrom(state.temp2, params.nvx * params.nvy);

	// projection on YZ plane
	reduce<value_type>(state.temp, state.temp2, params.cells, params.nvy * params.nvz);
	state.dens_a_zy.copyFrom(state.temp2, params.nvy * params.nvz);
	cutilSafeCall(transpose<value_type>(state.dens_a_zy, state.temp, params.nvy, params.nvz, 1));

	sparseReduce(state.temp, state.dens_b, params.cells * params.ne, params.cells);

	// projection on XY plane
	cutilSafeCall(transpose<value_type>(state.temp2, state.temp, params.nvx * params.nvy, params.nvz, 1));
	reduce<value_type>(state.temp2, state.dens_b, params.cells, params.nvx * params.nvy);
	state.dens_b_xy.copyFrom(state.temp2, params.nvx * params.nvy);

	// projection on YZ plane
	reduce<value_type>(state.temp, state.temp2, params.cells, params.nvy * params.nvz);
	state.dens_b_zy.copyFrom(state.temp2, params.nvy * params.nvz);
	cutilSafeCall(transpose<value_type>(state.dens_b_zy, state.temp, params.nvy, params.nvz, 1));
}

// Draw graphs from current state to provided buffers
void drawState(CalculationParameters &params, EvolutionState &state, CudaTexture &a_xy_tex,
	CudaTexture &b_xy_tex, CudaTexture &a_zy_tex, CudaTexture &b_zy_tex)
{

	value_type scale = 3.0 * params.N * params.ne / (params.cells * params.dx * params.dy * params.dz);
	value_type xy_scale = scale * params.nvz;
	value_type zy_scale = scale * params.nvx;

	drawData(a_xy_tex, state.dens_a_xy, xy_scale);
	drawData(b_xy_tex, state.dens_b_xy, xy_scale);
	drawData(a_zy_tex, state.dens_a_zy, zy_scale);
	drawData(b_zy_tex, state.dens_b_zy, zy_scale);

	float4 *b_zy_buf = b_zy_tex.map();
	cudaMemcpy(state.to_bmp, b_zy_buf, params.nvz * params.nvy * sizeof(float4), cudaMemcpyDeviceToHost);
	b_zy_tex.unmap();
}
