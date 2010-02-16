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

// initial parameters, real units
struct ModelParameters
{
	// scattering lengths, in Bohr radii
	value_type a11, a12, a22;

	value_type m; // mass of a particle
	value_type fx, fy, fz; // trap frequences, Hz

	int N; // number of particles

	value_type detuning; // detuning frequency, Hz
	value_type gamma111, gamma12, gamma22; // loss terms, cm^6/s for 111 and cm^3/s for others

	// set to 1 with vacuum noise, 0 otherwise
	value_type Va;
	value_type Vb;

	// number of points in space (must be power of 2 and greater than 16)
	int nvx;
	int nvy;
	int nvz;

	int itmax; // number of iterations for mid-step integral calculations

	// in seconds
	value_type dt_steady; // time step for steady-state calculation
	value_type t_equilib; // equilibration time
	value_type dt_evo; // evolution time step
	int ne; // number of samples in ensemble (must be power of 2)
};

// derived parameters, natural units
struct CalculationParameters
{
	int N;

	value_type g11; // nonlinear coupling in 1
	value_type g12; // nonlinear cross-coupling
	value_type g22; // nonlinear coupling in 2

	value_type mu; // chemical potential from TF approximation

	int cells; // number of space cells in one ensemble

	value_type Va;
	value_type Vb;
	value_type V; // average vacuum noise

	value_type lambda; // radial trap freq. / axial trap freq.
	value_type l_rho; // natural length
	value_type t_rho; // natural time

	value_type detuning;
	value_type l111, l12, l22; // loss terms, natural units

	// number of points in space (must be power of 2 and greater than 16)
	int nvx;
	int nvy;
	int nvz;

	// maximum coordinates
	value_type xmax;
	value_type ymax;
	value_type zmax;

	// space step
	value_type dx;
	value_type dy;
	value_type dz;

	// number of points in space in form of powers of 2
	// (to use in bitwise shifts instead of multiplication)
	int nvx_pow;
	int nvy_pow;
	int nvz_pow;

	// k step
	value_type dkx;
	value_type dky;
	value_type dkz;

	int itmax;

	value_type dt_steady;
	value_type t_equilib;
	value_type dt_evo;
	int ne;
};


struct EvolutionState
{
	batchfftHandle plan; // saved plan for FFTs

	CudaBuffer<value_pair> a, b, complex_t1, complex_t2; // current state vectors

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

	bool pi_pulse_applied;

	EvolutionState()
	{
		to_bmp = NULL;
		pi_pulse_applied = false;
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
		complex_t1.init(size);
		complex_t2.init(size);
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
		complex_t1.release();
		complex_t2.release();
	}

private:
	EvolutionState(const EvolutionState&);
	void operator=(const EvolutionState&);
};

#endif
