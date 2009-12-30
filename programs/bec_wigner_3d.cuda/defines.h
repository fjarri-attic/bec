#ifndef _DEFINES_H
#define _DEFINES_H

#include <cutil_inline.h>
#include "batchfft.h"
#include "cudabuffer.h"
#include "misc.h"

typedef float value_type;
typedef float2 value_pair;
#define MAKE_VALUE_PAIR make_float2

// number of maximum threads per block
// (must be a power of two)
#define MAX_THREADS_NUM 256

// maximum dimension of a grid
// (actually, maximum is 65535, but we use closest 2**n number for the sake of simplicity)
#define MAX_GRID_SIZE 32768

// size of half-warp (for kernel optimizations)
#define HALF_WARP_SIZE 16

struct CalculationParameters
{
	// in h-bar units
	value_type g11; // nonlinear coupling in 1
	value_type g12; // nonlinear cross-coupling
	value_type g22; // nonlinear coupling in 2
	value_type E; // energy splitting

	value_type m; // atom mass of a particle
	value_type fx, fy, fz; // trap frequences, Hz

	int N;

	value_type tscale; // time scale factor (ms)=1/omega_||

	// set to 1 with vacuum noise, 0 otherwise
	value_type Va;
	value_type Vb;

	// maximum coordinates, in meters
	value_type xmax;
	value_type ymax;
	value_type zmax;

	// number of points in space (must be power of 2 and greater than 16)
	int nvx;
	int nvy;
	int nvz;

	// steady state calculation parameters
	value_type tmaxGP; // maximum time for GP
	int itmax; // maximum iterations for GP
	value_type dtGP; // time step for GP
	int n0; // initial number per cell

	// evolution parameters
	value_type tmaxWig; // equilibration time
	value_type dtWig; // equilibration time step
	int ne; // number of samples in ensemble (must be power of 2)

// derived parameters

	value_type mu; // chemical potential

	int cells; // number of space cells in one ensemble

	value_type V; // average vacuum noise

	// space step
	value_type dx;
	value_type dy;
	value_type dz;

	// potential energy coefficients, h-bar units
	value_type px, py, pz;

	// number of points in space in form of powers of 2
	// (to use in bitwise shifts instead of multiplication)
	int nvx_pow;
	int nvy_pow;
	int nvz_pow;

	// k step
	value_type dkx;
	value_type dky;
	value_type dkz;

	value_type kcoeff;

};


struct EvolutionState
{
	batchfftHandle plan; // saved plan for FFTs

	CudaBuffer<value_pair> a, b; // current state vectors

	// temporary buffers for density calculations
	CudaBuffer<value_type> dens_a, dens_b, temp, temp2;

	// buffers for density graphs
	CudaBuffer<value_type> dens_a_xy, dens_b_xy, dens_a_zy, dens_b_zy;

	float4 *to_bmp;

	// kernel settings for calculations
	dim3 block;
	dim3 grid;

	dim3 xy_block, xy_grid, zy_block, zy_grid;

	value_type t;

	EvolutionState()
	{
		to_bmp = NULL;
	}

	EvolutionState(const CalculationParameters &params)
	{
		to_bmp = NULL;
		init(params);
	}

	~EvolutionState()
	{
		if(to_bmp != NULL)
			release();
	}

	void init(const CalculationParameters &params)
	{
		assert(to_bmp == NULL);
		size_t size = params.cells * params.ne;

		a.init(size);
		b.init(size);
		dens_a.init(size);
		dens_b.init(size);
		temp.init(size);
		temp2.init(size);
		dens_a_xy.init(params.nvx * params.nvy);
		dens_b_xy.init(params.nvx * params.nvy);
		dens_a_zy.init(params.nvz * params.nvy);
		dens_b_zy.init(params.nvz * params.nvy);

		to_bmp = new float4[params.cells];

		createKernelParams(block, grid, size, MAX_THREADS_NUM);
		createKernelParams(xy_block, xy_grid, params.nvx * params.nvy, MAX_THREADS_NUM);
		createKernelParams(zy_block, zy_grid, params.nvy * params.nvz, MAX_THREADS_NUM);

		cufftSafeCall(batchfftPlan3d(&plan, params.nvz, params.nvy, params.nvx, CUFFT_C2C, params.ne));

		t = 0;
	}

	void release()
	{
		assert(to_bmp != NULL);

		delete[] to_bmp;
		to_bmp = NULL;

		batchfftDestroy(plan);

		a.release();
		b.release();
		dens_a.release();
		dens_b.release();
		temp.release();
		temp2.release();
		dens_a_xy.release();
		dens_b_xy.release();
		dens_a_zy.release();
		dens_b_zy.release();
	}

private:
	EvolutionState(const EvolutionState&);
	void operator=(const EvolutionState&);
};

#endif
