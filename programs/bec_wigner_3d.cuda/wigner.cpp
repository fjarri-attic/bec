#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include <cutil_inline.h>
#include "defines.h"
#include "batchfft.h"
#include "bitmap.h"
#include "cudatexture.h"
#include "spectre.h"

#include <GL/glew.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

EvolutionState state;
CalculationParameters params;
CudaTexture a_xy, b_xy, a_zy, b_zy;

timeval time_start, time_current;
bool pause = false;

// CUDA functions
void initConstants(CalculationParameters &params);
void calculateSteadyState(value_pair *buf, CalculationParameters &params);
void initEvolution(value_pair *buf, CalculationParameters &params, EvolutionState &state);
void calculateEvolution(CalculationParameters &params, EvolutionState &state, value_type dt);
void drawState(CalculationParameters &params, EvolutionState &state, CudaTexture &a_xy_tex,
	CudaTexture &b_xy_tex, CudaTexture &a_zy_tex, CudaTexture &b_zy_tex);
void initWaveVectors(CalculationParameters &params);
void releaseWaveVectors();

// Initialize calculation
void fillCalculationParameters(CalculationParameters &params)
{
	params.N = 150000;

	params.m = 1.443160648e-25; // mass of one particle, kg (Rb, atom mass 87)

	// scattering lengths, in Bohr radii

	// from AS presentation
	//value_type a_11 = 100.44 * a0;
	//value_type a_22 = 95.47 * a0;
	//value_type a_12 = 98.09 * a0;

	// From "Spatially inhomogeneous phase evolution of a two-component Bose-Einstein condensate"
	//value_type a_11 = 100.40 * a0;
	//value_type a_22 = 95.00 * a0;
	//value_type a_12 = 97.66 * a0;

	params.a11 = 100.4;
	params.a22 = 95.0;
	params.a12 = 97.66;

	// Trap frequencies, Hz

	// From "Spatially inhomogeneous phase evolution of a two-component Bose-Einstein condensate"
	params.fx = 97.6;
	params.fy = 97.6;
	params.fz = 11.96;

	params.detuning = 41;
	params.gamma111 = 5.4e-30;
	params.gamma12 = 0.78e-13;
	params.gamma22 = 1.194e-13;

	// Vacuum noise, 0.0 - 1.0
	params.Va = 0;
	params.Vb = 0;

	// spatial lattice size
	params.nvx = 32;
	params.nvy = 32;
	params.nvz = 128;

	params.itmax = 3;

	params.dtGP = 0.01;
	params.tmaxWig = 2;
	params.dtWig = 0.01;

	params.ne = 1;
}

// auxiliary function - binary algorithm
// (2 -> 1, 4 -> 2, 8 -> 3 etc)
// works only for exact powers of 2
int log2(int input)
{
	int res = 1;

	if(input == 0 || input == 1)
		return 0;

	while(input != 2) {
		input >>= 1;
		res++;
	}

	return res;
}

// Derive dependent calculation parameters
void fillDerivedParameters(CalculationParameters &params)
{
	value_type h_bar = 1.054571628e-34;
	value_type a0 = 5.2917720859e-11; // Bohr radius, meters

	// natural units
	value_type w_rho = 2 * M_PI * params.fx;
	params.l_rho = sqrt(h_bar / (params.m * w_rho));
	params.t_rho = 1 / w_rho;
	params.lambda = params.fx / params.fz;

	params.cells = params.nvx * params.nvy * params.nvz;

	params.V = (params.Va + params.Vb) / 2.0;

	params.detuning_natural = params.detuning / w_rho;

	params.l111 = (params.gamma111 / 1e+12) / (pow(params.l_rho, 6) * w_rho);
	params.l12 = (params.gamma12 / 1e+6) / (pow(params.l_rho, 3) * w_rho);
	params.l22 = (params.gamma22 / 1e+6) / (pow(params.l_rho, 3) * w_rho);

	params.g11 = 4 * M_PI * params.a11 * a0 / params.l_rho;
	params.g12 = 4 * M_PI * params.a12 * a0 / params.l_rho;
	params.g22 = 4 * M_PI * params.a22 * a0 / params.l_rho;

	params.mu = pow(15.0 * params.N * params.g11 / (16.0 * M_PI * params.lambda * sqrt(2.0)), 0.4);
	printf("mu(TF) = %f\n", params.mu);

	params.xmax = 1.2 * sqrt(2 * params.mu);
	params.ymax = 1.2 * sqrt(2 * params.mu);
	params.zmax = 1.2 * params.lambda * sqrt(2 * params.mu);

	// space step
	params.dx = 2 * params.xmax / (params.nvx - 1);
	params.dy = 2 * params.ymax / (params.nvy - 1);
	params.dz = 2 * params.zmax / (params.nvz - 1);

	params.nvx_pow = log2(params.nvx);
	params.nvy_pow = log2(params.nvy);
	params.nvz_pow = log2(params.nvz);

	// k step
	params.dkx = M_PI / params.xmax;
	params.dky = M_PI / params.ymax;
	params.dkz = M_PI / params.zmax;
}

// Returns difference in seconds between to timevals
float time_diff(timeval &start, timeval &stop)
{
	return (stop.tv_sec - start.tv_sec) * 1.0f + (stop.tv_usec - start.tv_usec) / 1000000.0f;
}

// Idle function for glut application cycle
void idle(void)
{
	glutPostRedisplay();
}

// Keyboard events handler for glut application cycle
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch( key)
	{
	case 27:
		exit (0);
		break;
	case 32:
		pause = !pause;
		break;
	default:
		break;
	}
}

// Reshape events handler for glut application cycle
void reshape(int x, int y) {
	// set viewport size equal to window size
	glViewport(0, 0, x, y);

	// set orthographic projection with screen coordinates from (0, 0) to (1, 1)
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 1, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutPostRedisplay();
}

// Clean up buffers and resources
void cleanup(void) {
	a_xy.release();
	b_xy.release();
	a_zy.release();
	b_zy.release();

	// Free all CUDA resources
	state.release();
	releaseWaveVectors();
	releaseSpectre();
}

// Main on-paint handler
void display(void) {

	// propagate system
	if(!pause)
		calculateEvolution(params, state, params.dtWig);

	// fill vertex buffers with state graphs
	drawState(params, state, a_xy, b_xy, a_zy, b_zy);

	int t = (int)(state.t * params.t_rho * 1000.0); // time in ms

	if(t % 5 == 0)
	{
		char fname[255];
		sprintf(fname, "screenshot%03d.bmp", t);
		createBitmap(fname, state.to_bmp, params.nvz, params.nvy);
	}

	// draw graphs

	glEnable(GL_TEXTURE_2D);

	// clear screen
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	float sizex = params.xmax / (params.xmax + params.zmax);
	float sizez = params.zmax / (params.xmax + params.zmax);
	a_xy.draw(0, 0, sizex, 0.5);
	b_xy.draw(0, 0.5, sizex, 0.5);
	a_zy.draw(sizex, 0, sizez, 0.5);
	b_zy.draw(sizex, 0.5, sizez, 0.5);

	glutSwapBuffers();
	glutPostRedisplay();

	char title[256];
	sprintf(title, "Two-component BEC evolution: %d ms", t);
	glutSetWindowTitle(title);
}

// Initialize OpenGL subsystem
int initGL(int argc, char **argv, CalculationParameters &params)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	float width = params.xmax + params.zmax;
	float height = 2 * params.ymax;

	if(800.0 / height > 1000.0 / width)
		glutInitWindowSize(1000, (int)(1000.0 / width * height));
	else
		glutInitWindowSize((int)(800.0 / height * width), 800);

	glutCreateWindow("Two-component BEC evolution");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

	glewInit();
	if(!glewIsSupported("GL_ARB_vertex_buffer_object"))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return CUTFalse;
	}
	return CUTTrue;
}

int main(int argc, char** argv)
{
	fillCalculationParameters(params);
	fillDerivedParameters(params);

	if (CUTFalse == initGL(argc, argv, params))
		return CUTFalse;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if(cutCheckCmdLineFlag(argc, (const char**)argv, "device"))
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice(cutGetMaxGflopsDeviceId());

	// initialize calculations
	initConstants(params);

	timeval init_start, init_stop;

	// calculate steady state
	value_pair *steady_state = new value_pair[params.cells];

	initSpectre();
	initWaveVectors(params);

	gettimeofday(&init_start, NULL);
	calculateSteadyState(steady_state, params);
	gettimeofday(&init_stop, NULL);
	printf("Steady state calculation: %.3f s\n", time_diff(init_start, init_stop));

/*
	FILE *f = fopen("plot_gs_mu.txt", "w");
	int shift = (params.nvz / 2) * params.nvx * params.nvy + (params.nvy / 2) * params.nvx;
	for(int i = 0; i < params.nvx; i++)
	{
		value_pair val = steady_state[shift + i];
		fprintf(f, "%f %f\n", (-params.xmax + params.dx * i) * 1000000, (val.x * val.x + val.y * val.y));
	}
	fclose(f);
 */

	gettimeofday(&init_start, NULL);
	state.init(params);
	initEvolution(steady_state, params, state);
	gettimeofday(&init_stop, NULL);
	printf("Evolution init: %.3f s\n", time_diff(init_start, init_stop));

	delete[] steady_state;

	// measure propagation time, for testing purposes
	calculateEvolution(params, state, 0.0); // warm-up
	gettimeofday(&init_start, NULL);
	calculateEvolution(params, state, 0.0); // zero time step - because we are just measuring speed here
	gettimeofday(&init_stop, NULL);
	printf("Propagation time: %.3f ms\n", time_diff(init_start, init_stop) * 1000.0f);

	// prepare textures
	a_xy.init(params.nvx, params.nvy);
	b_xy.init(params.nvx, params.nvy);
	a_zy.init(params.nvz, params.nvy);
	b_zy.init(params.nvz, params.nvy);

	// remember starting time
	gettimeofday(&time_start, NULL);

	// start main application cycle
        atexit(cleanup);
        glutMainLoop();
	return 0;
}
