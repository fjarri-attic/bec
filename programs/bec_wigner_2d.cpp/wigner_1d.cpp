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
};

typedef vector< complex<double>, FFTWAlloc< complex<double> > > complex_vector;

template<class C1, class C2>
void save_results(const char *filename, vector<C1> &x, vector<C2> &y, const char *x_label, const char *y_label)
{
	ofstream s(filename, ios_base::trunc);
	s << x_label << " " << y_label << endl;
	for(int i = 0; i < x.size(); i++)
		s << x[i] << " " << y[i] << endl;
}

template<class C1, class C2, class C3>
void save_results(const char *filename, vector<C1> &x, vector<C2> &y, vector<C3> &z,
	const char *x_label, const char *y_label, const char *z_label)
{
	ofstream s(filename, ios_base::trunc);
	s << x_label << " " << y_label << " " << z_label << endl;
	for(int i = 0; i < x.size(); i++)
		s << x[i] << " " << y[i] << " " << z[i] << endl;
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
double zscale = 2.5;		//space scale factor (micron)?check!
double t0 = 54;			//decay time observed far from resonance
double Va = 0.5;
double Vb = 1;			//set to 1 with vacuum noise, 0 otherwise
double V = (Va + Vb) / 2.0;	//average vacuum noise

//////////////////////////////////////////////////////////////////////////
//	Sets up initial integration data values
//////////////////////////////////////////////////////////////////////////
double dt = .01;			//time step
double dz = 1.0;			//space step
double zmax = 20;		//maximum space coordinate
double dk = M_PI / zmax;		//k step
double tmaxGP = 1.0;		//maximum time for GP
int itmax = 3;			//maximum iterations for GP
double tmaxWig = 30.;		//maximum time for each Wigner step
int ntG = 1 + (int)(tmaxGP / dt);	//number of points in time
int ntW = 1 + (int)(tmaxWig / dt);	//number of points in time
int nv = 1 + 2 * (int)(zmax / dz);	//number of elements per vector
int ne = 20;			//number of samples in ensemble
int meas = 2;			//number of measurements per time-step
int n0 = 1;			//initial number per cell

// auxiliary values
double sdz = sqrt(1 / dz);		//square-root of space-step
double sdt = 1. / sqrt(dt);		//square-root of time-step
double dt2 = dt / 2.0;			//half of time-step
double r2 = sqrt(2.0);			//stores root-two

vector<double> prop, z, k, pG, pa, pb;
complex_vector propG, propW;

complex_vector a0;
vector<double> density;
double N0;

void fill_propagation_matrices()
{
	prop.resize(nv);
	z.resize(nv);
	k.resize(nv);
	pG.resize(nv);
	pa.resize(nv);
	pb.resize(nv);
	propG.resize(nv);
	propW.resize(nv);

	for(int j = 0; j < nv; j++)
	{
		z[j] = -zmax + dz * j;	//set up z coordinates
		k[j] = dk * j;		//set up +ive k coordinates
		if(2 * j > nv)
			k[j] = dk * (j - nv);	//set up -ive k coordinates

		prop[j] = -k[j] * k[j];			//set up k-space propagator
		pG[j] = mu - z[j] * z[j] / 4;		//set up GP x-space propagator

		pa[j] = mu - z[j] * z[j] / 4 + 0.5 * E;	//set up a-species x-space propagator
		pb[j] = mu - z[j] * z[j] / 4 - 0.5 * E;	//set up b-species x-space propagator
		z[j] *= zscale;				//scale z-coordinate for plotting

		propG[j] = exp(prop[j] * dt2);		//Linear propagate half of time-step
		propW[j] = exp(prop[j] * complex<double>(0, -dt2));	//Linear propagate half of time-step
	}
}

void calculate_steady_state()
{
	fftw_plan forward_plan, backward_plan;
	complex_vector a, a1, da;
	vector<double> t, ntot;

	a.resize(nv);
	a1.resize(nv);
	da.resize(nv);
	t.resize(ntG);
	ntot.resize(ntG);

	forward_plan = fftw_plan_dft_1d(nv, (fftw_complex*)&a[0], (fftw_complex*)&a[0],
		FFTW_FORWARD, FFTW_ESTIMATE);
	backward_plan = fftw_plan_dft_1d(nv, (fftw_complex*)&a[0], (fftw_complex*)&a[0],
		FFTW_BACKWARD, FFTW_ESTIMATE);

	//initial GP solution in k-space
	a.assign(nv, complex<double>(n0, 0));
	fftw_execute(backward_plan);
	for(int j = 0; j < nv; j++)
		a[j] /= nv;

	//////////////////////////////////////////////////////////////////////////
	// 	Starts GP loop in time: calculate mean-field steady-state
	//////////////////////////////////////////////////////////////////////////
	for(int n = 0; n < ntG; n++)		//loops until GP-time max
	{
		if(n > 0)			//First time only initializes a
		{
			for(int j = 0; j < nv; j++)
				a[j] = propG[j] * a[j];	//Linear propagate in k-space

			fftw_execute(forward_plan);	//FFT into x-space

			a1 = a;			//stores initial x-space field
			for(int iter = 0; iter < itmax; iter++)	//iterate to midpoint solution
				for(int j = 0; j < nv; j++)
				{
					da[j] = exp(dt2 * (pG[j] -
						g11 * conj(a[j]) * a[j])); //calculate midpoint log derivative and exponentiate
					a[j] = a1[j] * da[j];	//propagate to midpoint using log derivative
				}
						//end iterations to midpoint solution

			for(int j = 0; j < nv; j++)
				a[j] = a[j] * da[j];	//propagate to endpoint using log derivative

			fftw_execute(backward_plan);		//FFT into k-space
			for(int j = 0; j < nv; j++)
				a[j] /= nv;

			for(int j = 0; j < nv; j++)
				a[j] = propG[j] * a[j];		//Linear propagate in k-space
		}				//end initialize if

		t[n] = n * dt * tscale;		//Store t-coordinate

		//Store total number
		ntot[n] = 0;
		for(int j = 0; j < nv; j++)
			ntot[n] += abs(conj(a[j]) * a[j]);
		ntot[n] *= dz * nv;
	}					//end time loop

	fftw_execute(forward_plan);		//FFT into x-space
	a0 = a;					//Store GP x-space solution
	N0 = ntot[ntG - 1];

	density.resize(nv);
	for(int j = 0; j < nv; j++)
		density[j] = abs(conj(a[j]) * a[j]);

	save_results("plot5.dat", t, ntot, "Time", "Total number");

	fftw_destroy_plan(forward_plan);
	fftw_destroy_plan(backward_plan);
}

void main_loop()
{
	fftw_plan forward_plan_a, backward_plan_a, forward_plan_b, backward_plan_b, forward_plan_at;
	vector<double> t, avn, N, m, var;
	complex_vector a, b, da, db, at, a1, b1, b0;
	t.resize(ntW); //Initialise time
	a.resize(nv);
	b.resize(nv);
	at.resize(nv);
	m.resize(ntW * meas);
	var.resize(ntW * meas);
	avn.assign(nv, 0);
	da.resize(nv * 2);

	avn.assign(nv, 0);	//initialises average number in x-space
	N.assign(meas, 0);		//Initialises measurements
	m.assign(meas * ntW, 0);		//Initialises means
	var.assign(meas * ntW, 0);	//Initialises variances

	forward_plan_a = fftw_plan_dft_1d(nv, (fftw_complex*)&a[0], (fftw_complex*)&a[0],
		FFTW_FORWARD, FFTW_MEASURE);
	backward_plan_a = fftw_plan_dft_1d(nv, (fftw_complex*)&a[0], (fftw_complex*)&a[0],
		FFTW_BACKWARD, FFTW_MEASURE);
	forward_plan_b = fftw_plan_dft_1d(nv, (fftw_complex*)&b[0], (fftw_complex*)&b[0],
		FFTW_FORWARD, FFTW_MEASURE);
	backward_plan_b = fftw_plan_dft_1d(nv, (fftw_complex*)&b[0], (fftw_complex*)&b[0],
		FFTW_BACKWARD, FFTW_MEASURE);

	forward_plan_at = fftw_plan_dft_1d(nv, (fftw_complex*)&a[0], (fftw_complex*)&at[0],
		FFTW_FORWARD, FFTW_ESTIMATE);

	db.assign(nv, complex<double>(0, 0));		//vector with zeros everywhere
	b0.assign(nv, complex<double>(n0, 0));		//initialises vector amplitudes

	for(int j = 0; j < ne; j++) //loop over ensembles
	{
		//Initialises ns gaussian noises in x-space
		fill_with_normal_distribution(da, sdz / 2);

		for(int i = 0; i < nv; i++)
		{
			//Initialises a-ensemble amplitudes with vacuum noise
			a[i] = a0[i] + Va * da[i];

			//Initialises b-ensemble amplitudes with vacuum noise
			b[i] = Vb * da[nv + i];
		}

		fftw_execute(backward_plan_a);	//FFT into k-space
		fftw_execute(backward_plan_b);	//FFT into k-space
		for(int i = 0; i < nv; i++)
		{
			a[i] /= nv;
			b[i] /= nv;
		}

		//////////////////////////////////////////////////////////////////////////
		// Starts loop in time: first half is an equilibration phase, then interference
		//////////////////////////////////////////////////////////////////////////
		for(int n = -ntW; n < ntW; n++)		//loops until time tmax
		{
			if(n > -ntW) 		//First time only initializes a
			{
				if(n == 0)	//After equilibration time apply Bragg pulse
				{
					b0 = b;	//Store initial b-values

					//Apply pi/2 Bragg pulse
					for(int i = 0; i < nv; i++)
					{
						b[i] = (b[i] - complex<double>(0, 1) * a[i]) / r2;
						a[i] = (a[i] - complex<double>(0, 1) * b0[i]) / r2;
					}
				}					//End pulse if-statement

				//Linear propagate a,b-field
				for(int i = 0; i < nv; i++)
				{
					a[i] *= propW[i];
					b[i] *= propW[i];
				}

				//FFT into x-space
				fftw_execute(forward_plan_a);
				fftw_execute(forward_plan_b);

				for(int i = 0; i < nv; i++)
				{
					da[i] = -g11 * conj(a[i]) * a[i] - g12 * conj(b[i]) * b[i] + pa[i]; //calculate midpoint log derivative
					db[i] = -g22 * conj(b[i]) * b[i] - g12 * conj(a[i]) * a[i] + pb[i]; //calculate midpoint log derivative
					a[i] = a[i] * exp(complex<double>(0, -1) * dt * da[i]);	//Use log propagator to calculate next time point
					b[i] = b[i] * exp(complex<double>(0, -1) * dt * db[i]);	//Use log propagator to calculate next time point
				}

				fftw_execute(backward_plan_a);	//FFT into k-space
				fftw_execute(backward_plan_b);	//FFT into k-space
				for(int i = 0; i < nv; i++)
				{
					a[i] /= nv;
					b[i] /= nv;
				}

				//Linear propagate a,b-field
				for(int i = 0; i < nv; i++)
				{
					a[i] *= propW[i];
					b[i] *= propW[i];
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
				for(int i = 0; i < nv; i++)
				{
					b1[i] = (b[i] + complex<double>(0, 1) * a[i]) / r2;
					a1[i] = (a[i] + complex<double>(0, 1) * b[i]) / r2;
				}

				double sum_a1 = 0, sum_b1 = 0;
				for(int i = 0; i < nv; i++)
				{
					sum_a1 += abs(a1[i] * conj(a1[i]));
					sum_b1 += abs(b1[i] * conj(b1[i]));
				}

				double N1 = nv * (sum_a1 * dz - 0.5 * V);
				double N2 = nv * (sum_b1 * dz - 0.5 * V);

				//Mean amplitude over ensemble
				m[n] += N1;
				m[n + ntW] += N2;

				//Mean amplitude^2 over ensemble
				var[n] += N1 * N1;
				var[n + ntW] += N2 * N2;

				t[n] = n * dt * tscale;		//Store t-coordinate

				//cout << m[n] << "," << m[n+ntW] << "," << var[n]
				//	<< "," << var[n+ntW] << endl;
				//return;
			}
		} //end time loop

		fftw_execute(forward_plan_at); //transform back to x-space

		for(int i = 0; i < nv; i++)
			avn[i] += abs(at[i] * conj(at[i])) - 0.5 * V / dz;
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

	for(int i = 0; i < nv; i++)
		avn[i] = avn[i] / ne;

	save_results("plot6.dat", z, density, avn, "Z", "Number density", "Mean density");

	vector<double> temp1, temp2;
	temp1.resize(ntW);
	temp2.resize(ntW);

	for(int i = 0; i < ntW; i++)
	{
		temp1[i] = m[i];
		temp2[i] = N0 * .5 * (1 + exp(-t[i] / t0));
	}
	save_results("plot1.dat", t, temp1, temp2, "Time", "Mean number (a)", "");

	for(int i = 0; i < ntW; i++)
		temp1[i] = var[i];
	save_results("plot2.dat", t, temp1, "Time", "Variance (a)");

	for(int i = 0; i < ntW; i++)
		temp1[i] = m[i + ntW];
	save_results("plot3.dat", t, temp1, "Time", "Mean number (b)");

	for(int i = 0; i < ntW; i++)
		temp1[i] = var[i + ntW];
	save_results("plot4.dat", t, temp1, "Time", "Variance (b)");

	fftw_destroy_plan(forward_plan_a);
	fftw_destroy_plan(backward_plan_a);
	fftw_destroy_plan(forward_plan_b);
	fftw_destroy_plan(backward_plan_b);
	fftw_destroy_plan(forward_plan_at);
}

int main()
{
	fftw_plan p;
	srand(0);

	fill_propagation_matrices();

	calculate_steady_state();
	main_loop();

	return 0;
}
