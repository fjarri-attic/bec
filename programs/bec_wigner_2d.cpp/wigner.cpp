#include <complex>
#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

template<class T>
class FFTWAlloc
{
public:
	typedef size_t     size_type;
	typedef ptrdiff_t  difference_type;
	typedef T*       pointer;
	typedef const T* const_pointer;
	typedef T&       reference;
	typedef const T& const_reference;
	typedef T        value_type;

	size_type max_size() const throw()
	{
		return size_t(-1) / sizeof(T);
	}

	T* allocate(size_type n, const void* = 0)
	{
		T* ret = 0;
		if(n)
		{
			if(n <= this->max_size())
				ret = static_cast<T*>(fftw_malloc(n * sizeof(T)));
			else
				__throw_bad_alloc();
		}
		return ret;
	}

	void deallocate(pointer p, size_type n)
	{
		fftw_free(p);
	}

	void construct(pointer p, const T& val)
	{
		new(p) T(val);
	}

	void destroy(pointer p)
	{
		 p->~T();
	}

	template<typename T1>
	struct rebind
	{
		typedef FFTWAlloc<T1> other;
	};
};

typedef vector< complex<double>, FFTWAlloc< complex<double> > > complex_vector;

template<class C1, class C2>
void save_results(const char *filename, vector<C1> &x, vector<C2> &y)
{
	ofstream s(filename, ios_base::trunc);
	for(int i = 0; i < x.size(); i++)
		s << x[i] << " " << y[i] << endl;
}

template<class C1, class C2, class C3>
void save_results(const char *filename, vector<C1> &x, vector<C2> &y, vector<C3> &z)
{
	ofstream s(filename, ios_base::trunc);
	s << 0;
	for(int i = 0; i < x.size(); i++)
		s << " " << x[i];
	s << endl;

	for(int i = 0; i < y.size(); i++)
	{
		s << y[i];
		for(int j = 0; j < x.size(); j++)
			s << " " << z[i * x.size() + j];
		s << endl;
	}
}

double fill_with_normal_distribution(complex_vector &v, double dev)
{
	for(int i = 0; i < v.size(); i++)
	{
		double x1 = ((double)(rand() + 1))/((unsigned int)RAND_MAX + 1);
		double x2 = ((double)(rand() + 1))/((unsigned int)RAND_MAX + 1);

		double g1 = dev * sqrt(-2.0 * log(x1)) * cos(2 * M_PI * x2);
		double g2 = dev * sqrt(-2.0 * log(x1)) * sin(2 * M_PI * x2);

		v[i] = complex<double>(g1, g2);
	}
}

double scale = 1.0;		//nonlinear scale factor
double g11 = 6.18 * scale;	//nonlinear coupling in 1
double g12 = 6.03 * scale;	//nonlinear cross-coupling
double g22 = 5.88 * scale;	//nonlinear coupling in 2
double E = 2.500;		//energy splitting
double mu = 26.8 * sqrt(scale);	//chemical potential
double tscale = 1.76;		//time scale factor (ms)=1/omega_||
double space_scale = 2.5;	//space scale factor (micron)?check!
double t0 = 54;			//decay time observed far from resonance
double Va = 0.2;
double Vb = 0.5;		//set to 1 with vacuum noise, 0 otherwise
double V = (Va + Vb) / 2.0;	//average vacuum noise

//////////////////////////////////////////////////////////////////////////
//	Sets up initial integration data values
//////////////////////////////////////////////////////////////////////////
double xmax = 20, ymax = 20;	//maximum space coordinate
double dkx = M_PI / xmax, dky = M_PI / ymax;	//k step
double tmaxGP = 1.0;		//maximum time for GP
int itmax = 3;			//maximum iterations for GP
double tmaxWig = 2.0;		//maximum time for each Wigner step
int ntG = 32;	//number of points in time
double dtG = tmaxGP / (ntG - 1);			//time step
double dtW = 0.01;
int ntW = (int)(tmaxWig / dtW);
int nvx = 128;	//number of elements per vector, X coordinate
int nvy = 128;	//number of elements per vector, Y coordinate
double dx = 2.0 * xmax / (nvx - 1), dy = 2.0 * ymax / (nvy - 1); //space step
int ne = 32;			//number of samples in ensemble
int meas = 2;			//number of measurements per time-step
int n0 = 1;			//initial number per cell

// auxiliary values
double sdx = sqrt(1 / dx);
double sdy = sqrt(1 / dy);		//square-root of space-step
double sdtG = 1. / sqrt(dtG);		//square-root of time-step
double sdtW = 1. / sqrt(dtW);		//square-root of time-step
double dtG2 = dtG / 2.0;			//half of time-step
double dtW2 = dtW / 2.0;			//half of time-step
double r2 = sqrt(2.0);			//stores root-two

vector<double> prop, x, y, kx, ky, pG, pa, pb;
complex_vector propG, propW;

complex_vector a0;
vector<double> density;
double N0;

void fill_propagation_matrices()
{
	prop.resize(nvx * nvy);
	x.resize(nvx);
	y.resize(nvy);
	kx.resize(nvx);
	ky.resize(nvy);
	pG.resize(nvx * nvy);
	pa.resize(nvx * nvy);
	pb.resize(nvx * nvy);
	propG.resize(nvx * nvy);
	propW.resize(nvx * nvy);

	for(int i = 0; i < nvx; i++)
	{
		x[i] = -xmax + dx * i;	//set up coordinates
		kx[i] = (2 * i > nvx) ? dkx * (i - nvx) : dkx * i;

		for(int j = 0; j < nvy; j++)
		{
			y[j] = -ymax + dy * j;	//set up coordinates
			ky[j] = (2 * j > nvy) ? dky * (j - nvy) : dky * j;

			int index = j * nvx + i;
			double radius = x[i] * x[i] + y[j] * y[j];

			prop[index] = - kx[i] * kx[i] - ky[j] * ky[j]; //set up k-space propagator
			pG[index] = mu - radius / 4; //set up GP x-space propagator

			pa[index] = mu - radius / 4 + 0.5 * E;	//set up a-species x-space propagator
			pb[index] = mu - radius / 4 - 0.5 * E;	//set up b-species x-space propagator

			y[j] *= space_scale;	//scale y-coordinate for plotting

			propG[index] = exp(prop[index] * dtG2); //Linear propagate half of time-step
			propW[index] = exp(prop[index] * complex<double>(0, -dtW2)); //Linear propagate half of time-step
		}

		x[i] *= space_scale;	//scale x-coordinate for plotting
	}
}

void calculate_steady_state()
{
	fftw_plan forward_plan, backward_plan;
	complex_vector a, a1, da;
	vector<double> t, ntot;

	a.resize(nvx * nvy);
	a1.resize(nvx * nvy);
	da.resize(nvx * nvy);
	t.resize(ntG);
	ntot.resize(ntG);

	forward_plan = fftw_plan_dft_2d(nvx, nvy, (fftw_complex*)&a[0], (fftw_complex*)&a[0],
		FFTW_FORWARD, FFTW_ESTIMATE);
	backward_plan = fftw_plan_dft_2d(nvx, nvy, (fftw_complex*)&a[0], (fftw_complex*)&a[0],
		FFTW_BACKWARD, FFTW_ESTIMATE);

	//initial GP solution in k-space
	a.assign(nvx * nvy, complex<double>(n0, 0));
	fftw_execute(backward_plan);
	for(int s = 0; s < nvx * nvy; s++)
		a[s] /= nvx * nvy;

	//////////////////////////////////////////////////////////////////////////
	// 	Starts GP loop in time: calculate mean-field steady-state
	//////////////////////////////////////////////////////////////////////////
	for(int n = 0; n < ntG; n++)		//loops until GP-time max
	{
		if(n > 0) //First time only initializes a
		{
			for(int s = 0; s < nvx * nvy; s++)
				a[s] *= propG[s]; //Linear propagate in k-space

			fftw_execute(forward_plan); //FFT into x-space

			a1 = a; //stores initial x-space field
			for(int iter = 0; iter < itmax; iter++)	//iterate to midpoint solution
			{
				for(int s = 0; s < nvx * nvy; s++)
				{
					da[s] = exp(dtG2 * (pG[s] -
						g11 * conj(a[s]) * a[s])); //calculate midpoint log derivative and exponentiate
					a[s] = a1[s] * da[s];	//propagate to midpoint using log derivative
				}
			}

			for(int s = 0; s < nvx * nvy; s++)
				a[s] *= da[s]; //propagate to endpoint using log derivative

			fftw_execute(backward_plan); //FFT into k-space
			for(int s = 0; s < nvx * nvy; s++)
				a[s] /= nvx * nvy;

			for(int s = 0; s < nvx * nvy; s++)
				a[s] *= propG[s]; //Linear propagate in k-space
		} //end initialize if

		t[n] = n * dtG * tscale; //Store t-coordinate

		//Store total number
		ntot[n] = 0;
		for(int s = 0; s < nvx * nvy; s++)
			ntot[n] += abs(conj(a[s]) * a[s]);
		ntot[n] *= dx * dy * nvx * nvy;
	}					//end time loop

	fftw_execute(forward_plan);		//FFT into x-space
	a0 = a;					//Store GP x-space solution
	N0 = ntot[ntG - 1];

	density.resize(nvx * nvy);
	for(int s = 0; s < nvx * nvy; s++)
		density[s] = abs(conj(a[s]) * a[s]);

	save_results("plot5.dat", t, ntot);
	save_results("plot6-1.dat", x, y, density);

	fftw_destroy_plan(forward_plan);
	fftw_destroy_plan(backward_plan);
}


void main_loop()
{
	fftw_plan forward_plan_a, backward_plan_a, forward_plan_b, backward_plan_b, forward_plan_at;
	vector<double> t, avn, N, m, var;
	complex_vector a, b, da, db, at, a1, b1, b0;
	t.resize(ntW); //Initialise time
	a.resize(nvx * nvy);
	b.resize(nvx * nvy);
	at.resize(nvx * nvy);
	m.resize(ntW * meas);
	var.resize(ntW * meas);
	avn.assign(nvx * nvy, 0);
	da.resize(nvx * nvy * 2);

	avn.assign(nvx * nvy, 0); //initialises average number in x-space
	N.assign(meas, 0); //Initialises measurements
	m.assign(meas * ntW, 0); //Initialises means
	var.assign(meas * ntW, 0); //Initialises variances

	forward_plan_a = fftw_plan_dft_2d(nvx, nvy, (fftw_complex*)&a[0], (fftw_complex*)&a[0],
		FFTW_FORWARD, FFTW_MEASURE);
	backward_plan_a = fftw_plan_dft_2d(nvx, nvy, (fftw_complex*)&a[0], (fftw_complex*)&a[0],
		FFTW_BACKWARD, FFTW_MEASURE);
	forward_plan_b = fftw_plan_dft_2d(nvx, nvy, (fftw_complex*)&b[0], (fftw_complex*)&b[0],
		FFTW_FORWARD, FFTW_MEASURE);
	backward_plan_b = fftw_plan_dft_2d(nvx, nvy, (fftw_complex*)&b[0], (fftw_complex*)&b[0],
		FFTW_BACKWARD, FFTW_MEASURE);

	forward_plan_at = fftw_plan_dft_2d(nvx, nvy, (fftw_complex*)&a[0], (fftw_complex*)&at[0],
		FFTW_FORWARD, FFTW_ESTIMATE);

	db.assign(nvx * nvy, complex<double>(0, 0)); //vector with zeros everywhere
	b0.assign(nvx * nvy, complex<double>(n0, 0)); //initialises vector amplitudes

	for(int j = 0; j < ne; j++) //loop over ensembles
	{
		//Initialises ns gaussian noises in x-space
		fill_with_normal_distribution(da, 1 / sqrt(dx * dy));

		for(int s = 0; s < nvx * nvy; s++)
		{
			//Initialises a-ensemble amplitudes with vacuum noise
			a[s] = a0[s] + Va * da[s];

			//Initialises b-ensemble amplitudes with vacuum noise
			b[s] = Vb * da[nvx * nvy + s];
		}

		fftw_execute(backward_plan_a);	//FFT into k-space
		fftw_execute(backward_plan_b);	//FFT into k-space
		for(int s = 0; s < nvx * nvy; s++)
		{
			a[s] /= nvx * nvy;
			b[s] /= nvx * nvy;
		}

		//////////////////////////////////////////////////////////////////////////
		// Starts loop in time: first half is an equilibration phase, then interference
		//////////////////////////////////////////////////////////////////////////
		for(int n = -ntW; n <= 0; n++)		//loops until time tmax
		{
			if(n > -ntW) 		//First time only initializes a
			{
				if(n == 0)	//After equilibration time apply Bragg pulse
				{
					b0 = b;	//Store initial b-values

					//Apply pi/2 Bragg pulse
					for(int s = 0; s < nvx * nvy; s++)
					{
						b[s] = (b[s] - complex<double>(0, 1) * a[s]) / r2;
						a[s] = (a[s] - complex<double>(0, 1) * b0[s]) / r2;
					}


				}					//End pulse if-statement

				//Linear propagate a,b-field
				for(int s = 0; s < nvx * nvy; s++)
				{
					a[s] *= propW[s];
					b[s] *= propW[s];
				}

				//FFT into x-space
				fftw_execute(forward_plan_a);
				fftw_execute(forward_plan_b);

				for(int s = 0; s < nvx * nvy; s++)
				{
					da[s] = -g11 * conj(a[s]) * a[s] - g12 * conj(b[s]) * b[s] + pa[s]; //calculate midpoint log derivative
					db[s] = -g22 * conj(b[s]) * b[s] - g12 * conj(a[s]) * a[s] + pb[s]; //calculate midpoint log derivative
					a[s] = a[s] * exp(complex<double>(0, -1) * dtW * da[s]);	//Use log propagator to calculate next time point
					b[s] = b[s] * exp(complex<double>(0, -1) * dtW * db[s]);	//Use log propagator to calculate next time point
				}

				fftw_execute(backward_plan_a);	//FFT into k-space
				fftw_execute(backward_plan_b);	//FFT into k-space
				for(int s = 0; s < nvx * nvy; s++)
				{
					a[s] /= nvx * nvy;
					b[s] /= nvx * nvy;
				}

				//Linear propagate a,b-field
				for(int s = 0; s < nvx * nvy; s++)
				{
					a[s] *= propW[s];
					b[s] *= propW[s];
				}
			}	//end initialize if

			//////////////////////////////////////////////////////////////////////////
			// Calculates interference pattern
			//////////////////////////////////////////////////////////////////////////
			if(n >= 0)	//Store data for plotting
			{

				b1 = b;
				a1 = a;

				//Second Bragg pulse
				for(int s = 0; s < nvx * nvy; s++)
				{
					b1[s] = (b[s] + complex<double>(0, 1) * a[s]) / r2;
					a1[s] = (a[s] + complex<double>(0, 1) * b[s]) / r2;
				}

				double sum_a1 = 0, sum_b1 = 0;
				for(int s = 0; s < nvx * nvy; s++)
				{
					sum_a1 += abs(a1[s] * conj(a1[s]));
					sum_b1 += abs(b1[s] * conj(b1[s]));
				}

				double N1 = nvx * nvy * (sum_a1 * dx * dy - 0.5 * V);
				double N2 = nvx * nvy * (sum_b1 * dx * dy - 0.5 * V);

				//Mean amplitude over ensemble
				m[n] += N1;
				m[n + ntW] += N2;

				//Mean amplitude^2 over ensemble
				var[n] += N1 * N1;
				var[n + ntW] += N2 * N2;

				t[n] = n * dtW * tscale;		//Store t-coordinate
			}
		} //end time loop

		fftw_execute(forward_plan_at); //transform back to x-space

		for(int s = 0; s < nvx * nvy; s++)
			avn[s] += abs(at[s] * conj(at[s])) - 0.5 * V / dx / dy;

	} //end ensemble loop

	//////////////////////////////////////////////////////////////////////////
	// Outputs final mean and variance
	//////////////////////////////////////////////////////////////////////////
	for(int i = 0; i < ntW * 2; i++)
	{
		m[i] = m[i] / ne; //Mean at end of loop
		var[i] = (var[i] / ne - m[i] * m[i]);
		var[i] = sqrt(var[i] / (ne - 1)); //SD of mean at end of loop
	}

	for(int s = 0; s < nvx * nvy; s++)
		avn[s] /= ne;

	save_results("plot6-2.dat", x, y, avn);

	vector<double> temp;
	temp.resize(ntW);

	// mean number (a)
	for(int i = 0; i < ntW; i++)
		temp[i] = N0 * .5 * (1 + exp(-t[i] / t0));
	save_results("plot1-1.dat", t, m);
	save_results("plot1-2.dat", t, temp);

	// variance (a)
	save_results("plot2.dat", t, var);

	// mean number (b)
	for(int i = 0; i < ntW; i++)
		temp[i] = m[i + ntW];
	save_results("plot3.dat", t, temp);

	// variance (b)
	for(int i = 0; i < ntW; i++)
		temp[i] = var[i + ntW];
	save_results("plot4.dat", t, temp);

	fftw_destroy_plan(forward_plan_a);
	fftw_destroy_plan(backward_plan_a);
	fftw_destroy_plan(forward_plan_b);
	fftw_destroy_plan(backward_plan_b);
	fftw_destroy_plan(forward_plan_at);
}

int main()
{
	fftw_plan p;
	srand(time(0));


	fill_propagation_matrices();

//	fftw_init_threads();
//	fftw_plan_with_nthreads(1);
	calculate_steady_state();
	main_loop();
//	fftw_cleanup_threads();

	return 0;
}
