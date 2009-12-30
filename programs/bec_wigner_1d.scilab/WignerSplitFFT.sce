//////////////////////////////////////////////////////////////////////////
// 	Wigner interferometry SOLVER USING SCILAB
//
//	Number conserving split-step routine with FFT
//
//////////////////////////////////////////////////////////////////////////
clear;			//clears and initialises

//////////////////////////////////////////////////////////////////////////
//	Sets up initial science data values
//////////////////////////////////////////////////////////////////////////

scale = 1.0;		//nonlinear scale factor
g11 = 6.18 * scale;	//nonlinear coupling in 1
g12 = 6.03 * scale;	//nonlinear cross-coupling
g22 = 5.88 * scale;	//nonlinear coupling in 2
N = 0;			//number of atoms
E = 2.500;		//energy splitting
mu = 26.8 * sqrt(scale);	//chemical potential
tscale = 1.76;		//time scale factor (ms)=1/omega_||
zscale = 2.5;		//space scale factor (micron)?check!
t0 = 54;		//decay time observed far from resonance
Va = 0.5;
Vb = 1;			//set to 1 with vacuum noise, 0 otherwise
V = (Va + Vb) / 2.0;	//average vacuum noise

//////////////////////////////////////////////////////////////////////////
//	Sets up initial integration data values
//////////////////////////////////////////////////////////////////////////
dt = .01;		//time step
dz = 1.0;		//space step
zmax = 20;		//maximum space coordinate
dk = %pi / zmax;		//k step
tmaxGP = 1.0;		//maximum time for GP
itmax = 3;		//maximum iterations for GP
tmaxWig = 30.;		//maximum time for each Wigner step
ntG = 1 + tmaxGP / dt;	//number of points in time
ntW = 1 + tmaxWig / dt;	//number of points in time
nv = 1 + 2. * zmax / dz;	//number of elements per vector
meas = 2;		//number of measurements
ne = 20;		//number of samples in ensemble
meas = 2;		//number of measurements per time-step
n0 = 1;			//initial number per cell

//////////////////////////////////////////////////////////////////////////
//	Sets up propagation matrix
//////////////////////////////////////////////////////////////////////////
prop=zeros(nv);				//initial k-propagation matrix
for j = 1:nv;				//loop over space variables
	z(j) = -zmax + (j - 1) * dz;	//set up z coordinates
	k(j) = dk * (j - 1);		//set up +ive k coordinates
	if 2 * (j - 1) > nv then
		k(j) = dk * (j - 1 - nv);	//set up -ive k coordinates
	end;
	prop(j) = -k(j) ^ 2;			//set up k-space propagator
	pG(j) = mu - z(j) ^ 2 / 4;		//set up GP x-space propagator
	pa(j) = mu - z(j) ^ 2 / 4 + 0.5 * E;	//set up a-species x-space propagator
	pb(j) = mu - z(j) ^ 2 / 4 - 0.5 * E;	//set up b-species x-space propagator
	z(j) = z(j) * zscale;			//scale z-coordinate for plotting
end;

//////////////////////////////////////////////////////////////////////////
//	Sets up integration vectors
//////////////////////////////////////////////////////////////////////////
da = 0.0*%i + zeros(nv, 1);	//vector with zeros everywhere
a = zeros(nv, 1) + n0 + 0.0*%i;	//initialises vector amplitudes in x-space
avn = zeros(nv, 1);		//initialises average number in x-space
a = fft(a, 1);			//initial GP solution in k-space
db = 0.0*%i + zeros(nv,1);	//vector with zeros everywhere
b0 = zeros(nv, 1) + n0 + 0.0*%i;	//initialises vector amplitudes
sdz = sqrt(1 / dz);		//square-root of space-step
sdt = 1. / sqrt(dt);		//square-root of time-step
dt2 = dt / 2.0;			//half of time-step
r2 = sqrt(2);			//stores root-two
propG = exp(dt2 * prop);		//Linear propagate half of time-step
propW = exp(-%i * dt2 * prop);	//Linear propagate half of time-step
N = zeros(meas, 1);		//Initialises measurements
m = zeros(meas, ntW);		//Initialises means
var = zeros(meas, ntW);		//Initialises variances
t = zeros(1, ntG);		//Initialise time

//////////////////////////////////////////////////////////////////////////
// 	Starts GP loop in time: calculate mean-field steady-state
//////////////////////////////////////////////////////////////////////////
for n = 1:ntG;				//loops until GP-time max
	if n > 1 then			//First time only initializes a
		a = propG .* a;		//Linear propagate in k-space
		a = fft(a);		//FFT into x-space
		a1 = a;			//stores initial x-space field
		for iter = 1,itmax;	//iterate to midpoint solution
			da = exp(dt2 * (pG - g11 * conj(a) .* a)); //calculate midpoint log derivative and exponentiate
			a = a1 .* da;	//propagate to midpoint using log derivative
		end;			//end iterations to midpoint solution
		a = a .* da;		//propagate to endpoint using log derivative
		a = fft(a, 1);		//FFT into k-space
		a = propG .* a;		//Linear propagate in k-space
	end;				//end initialize if
	t(n) = (n - 1) * dt * tscale;		//Store t-coordinate
	ntot(n) = sum(a .* conj(a)) * dz * nv;	//Store total number
end;					//end time loop
a = fft(a);				//FFT into x-space
a0 = a;					//Store GP x-space solution
N0 = ntot(ntG);

//////////////////////////////////////////////////////////////////////////
//    GP Graphics section: plot mean-field steady-state
//////////////////////////////////////////////////////////////////////////
fs = 4;
//scf(5);
//plot(t, ntot, '-'); //Plot total number vs time
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('N', 'FontSize', fs);
//scf(6);
//plot(z, a .* conj(a), '-');//Plot number density vs space
//xlabel('z (micron)', 'FontSize', fs);
//ylabel('N', 'FontSize', fs);

//////////////////////////////////////////////////////////////////////////
// Starts loop over stochastic initial value ensembles
//////////////////////////////////////////////////////////////////////////
t = zeros(1, ntW); //Initialise time
a_f = zeros(nv * ne, 1);
b_f = zeros(nv * ne, 1);

for j = 1:ne;
  stt = 1 + (j - 1) * nv;
  stp = j * nv;

  da = 0.5 * grand(nv, 4, 'nor', 0.0, sdz);  
  a_f(stt:stp) = a0 + Va * (da(:, 1) + %i * da(:, 2));
  b_f(stt:stp) = Vb * (da(:, 3) + %i * da(:, 4));
  
  b_f(stt:stp)=fft(b_f(stt:stp), 1);	//FFT into k-space
  a_f(stt:stp)=fft(a_f(stt:stp), 1);	//FFT into k-space
end;

	//////////////////////////////////////////////////////////////////////////
	// Starts loop in time: first half is an equilibration phase, then interference
	//////////////////////////////////////////////////////////////////////////
for n = 1-ntW:ntW;		//loops until time tmax
  avn = zeros(nv, 1);		//initialises average number in x-space
  
  	for j = 1:ne; //loop over ensembles
    	
                  stt = 1 + (j - 1) * nv;
                  stp = j * nv;
                  
                  a = a_f(stt:stp)
                  b = b_f(stt:stp)
      	
    		if n > 1 - ntW then		//First time only initializes a
			if n == 1 then	//After equilibration time apply Bragg pulse
				b0 = b;	//Store initial b-values
				b = (b - %i*a) / r2; a = (a - %i*b0) / r2;	//Apply pi/2 Bragg pulse
			end;					//End pulse if-statement
			a = propW .* a; b = propW .* b;		//Linear propagate a,b-field
			a = fft(a); b = fft(b);			//FFT into x-space
			da = -g11 * conj(a) .* a - g12 * conj(b) .* b + pa; //calculate midpoint log derivative
			db = -g22 * conj(b) .* b - g12 * conj(a) .* a + pb; //calculate midpoint log derivative
			a = a .* exp(-%i * dt * da(:,1));	//Use log propagator to calculate next time point
			b = b .* exp(-%i * dt * db(:,1));	//Use log propagator to calculate next time point
			a = fft(a, 1); b = fft(b, 1);		//transform back to k-space
			a = propW .* a; b = propW .* b;		//Linear propagate a,b-field
		end;	//end initialize if

		//////////////////////////////////////////////////////////////////////////
		// Calculates interference pattern
		//////////////////////////////////////////////////////////////////////////
		if n >= 1 then	//Store data for plotting
			b1 = b; a1 = a;
			b1 = (b + %i*a) / r2; a1 = (a + %i*b) / r2; //Second Bragg pulse
			N(1) = nv * (sum(a1 .* conj(a1)) * dz - 0.5 * V);
			N(2) = nv * (sum(b1 .* conj(b1)) * dz - 0.5 * V);
			m(:, n) = N + m(:, n);			//Mean amplitude over ensemble
			var(:, n) = var(:, n) + N .^ 2;		//Mean amplitude^2 over ensemble
			t(n) = (n - 1) * dt * tscale;		//Store t-coordinate

                  end;
                  a_f(stt:stp) = a;
                  b_f(stt:stp) = b;
                  
                  at = fft(a);
            	avn = at .* conj(at) - 0.5 * V / dz + avn;
	                  

	end; //end time loop

            		
	scf(7);
	clf(7);
	plot(z, avn, '-');
					
end; //end ensemble loop

//////////////////////////////////////////////////////////////////////////
// Outputs final mean and variance
//////////////////////////////////////////////////////////////////////////
m = m / ne; //Mean at end of loop
var = (var / ne - m .^ 2);
var = sqrt(var / (ne - 1)); //SD of mean at end of loop
avn = avn / ne;

//////////////////////////////////////////////////////////////////////////
// Graphics section
//////////////////////////////////////////////////////////////////////////
fs = 4;
//scf(6);
//plot(z, avn, '-'); //Plot mean density
//xlabel('z (micron)', 'FontSize', fs);
//ylabel('N', 'FontSize', fs);

//////////////////////////////////////////////////////////////////////////
//scf(1);
//plot(t, m(1,:), t, N0 * .5 * (1 + exp(-t / t0)), '-'); //Plot mean number (a)
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('N1', 'FontSize', fs);
//scf(2);
//plot(t, var(1,:), '-'); //Plot  variance
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('<Delta N1>', 'FontSize', fs);

//////////////////////////////////////////////////////////////////////////
//scf(3);
//plot(t, m(2,:), '-'); //Plot  mean number (b)
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('N2', 'FontSize', fs);
//scf(4);
//plot(t, var(2,:), '-'); //Plot variance
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('<Delta N2>', 'FontSize', fs);
