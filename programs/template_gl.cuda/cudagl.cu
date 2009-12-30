/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <cufft.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

void cleanup(void);

GLuint gltex;
GLuint vbo;

int nvx = 512, nvy = 512;
int wWidth = nvx, wHeight = nvy;

texture<float4, 1> rainbow_tex;
cudaArray *rainbow_arr;

void setupTexture() {
	float4 *h_colors = (float4*)malloc(1530 * sizeof(float4));

	int R, G, B;

	for (int i = 0; i < 1530; i++)
	{
		if (i <= 255)
		// black -> violet
		{
			B = i;
			G = 0;
			R = i;
		}
		else if (i <= 255 * 2)
		// violet -> blue
		{
			B = 255;
			G = 0;
			R = 255 * 2 - i;
		}
		else if (i <= 255 * 3)
		// blue -> teal
		{
			B = 255;
			G = i - 255 * 2;
			R = 0;
		}
		else if (i <= 255 * 4)
		// teal -> green
		{
			B = 255 * 4 - i;
			G = 255;
			R = 0;
		}
		else if (i <= 255 * 5)
		// green -> yellow
		{
			B = 0;
			G = 255;
			R = i - 255 * 4;
		}
		else if (i <= 255 * 6)
		// yellow -> red
		{
			B = 0;
			G = 255 * 6 - i;
			R = 255;
		}

		h_colors[i].x = (float)R / 256;
		h_colors[i].y = (float)G / 256;
		h_colors[i].z = (float)B / 256;
		h_colors[i].w = 1.0;
	}

	rainbow_tex.filterMode = cudaFilterModeLinear;
	rainbow_tex.normalized = true;
	rainbow_tex.addressMode[0] = cudaAddressModeClamp;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();

	cudaMallocArray(&rainbow_arr, &desc, 1530, 1);
	cutilCheckMsg("cudaMalloc failed");

	cudaMemcpyToArray(rainbow_arr, 0, 0, h_colors, 1530 * sizeof(float4), cudaMemcpyHostToDevice);
	cutilCheckMsg("cudaMemcpy failed");

	cudaBindTextureToArray(rainbow_tex, rainbow_arr, desc);
	cutilCheckMsg("cudaBindTexture failed");

	free(h_colors);
}

void deleteTexture()
{
	cudaUnbindTexture(rainbow_tex);
	cudaFreeArray(rainbow_arr);
}

__global__ void fillBuf(float4 *buf, int nvx, int nvy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

	float y = (float)(tid / nvx);
	float x = (float)(tid % nvx);

	buf[tid] = tex1D(rainbow_tex, x * y / (nvx * nvy));
}

void display(void) {

	// perform CUDA steps
	dim3 grid(nvx * nvy / 512, 1, 1);
	dim3 tids(512, 1, 1);
	float4 *p;

	cudaGLMapBufferObject((void**)&p, vbo);
	cutilCheckMsg("cudaGLMapBufferObject failed");

	fillBuf<<<grid, tids>>>(p, nvx, nvy);
	cutilCheckMsg("advectParticles_k failed.");

	cudaGLUnmapBufferObject(vbo);
	cutilCheckMsg("cudaGLUnmapBufferObject failed");

	// draw textured polygon
	glEnable(GL_TEXTURE_2D);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	glBindTexture(GL_TEXTURE_2D, gltex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nvx, nvy, GL_RGBA, GL_FLOAT, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0, 0.0 );
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0, 0.0 );
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0, 1.0 );
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0, 1.0 );
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// Finish timing before swap buffers to avoid refresh sync
	glutSwapBuffers();

	glutPostRedisplay();
}

void idle(void) {
    glutPostRedisplay();
}

void keyboard( unsigned char key, int x, int y) {
    switch( key) {
        case 27: exit (0); break;
        default: break;
    }
}

void reshape(int x, int y) {
	wWidth = x; wHeight = y;
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 1, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
}

void cleanup(void) {
	cutilSafeCall(cudaGLUnregisterBufferObject(vbo));

	// delete vertex buffer
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glDeleteBuffersARB(1, &vbo);

	// delete openGL texture
	glBindTexture(GL_TEXTURE_2D, 0);
	glDeleteTextures(1, &gltex);

	deleteTexture();
}

int initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("Compute Stable Fluids");
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
	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (CUTFalse == initGL(argc, argv))
		return CUTFalse;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilGLDeviceInit(argc, argv);
	else
		cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());

	// prepare texture
	glGenTextures(1, &gltex);
	glBindTexture(GL_TEXTURE_2D, gltex);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, nvx, nvy, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glBindTexture(GL_TEXTURE_2D, 0);

	// prepare vertex buffer
	glGenBuffersARB(1, &vbo);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(float4) * nvx * nvy, NULL, GL_DYNAMIC_DRAW_ARB);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	cutilSafeCall(cudaGLRegisterBufferObject(vbo));
	setupTexture();

	atexit(cleanup);
	glutMainLoop();

	cudaThreadExit();
	return 0;
}
