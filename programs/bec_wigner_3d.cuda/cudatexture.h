#ifndef _CUDATEXTURE_H
#define _CUDATEXTURE_H

#include <GL/glew.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>

#include <assert.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

class CudaTexture
{
	GLuint vbo;
	GLuint gltex;

	int width;
	int height;

	bool initialized;

	CudaTexture(const CudaTexture&) {}
	void operator=(const CudaTexture&) {}

public:
	CudaTexture(): initialized(false) {}

	void init(int w, int h)
	{
		assert(!initialized);

		width = w;
		height = h;

		glGenTextures(1, &gltex);
		glBindTexture(GL_TEXTURE_2D, gltex);
		glTexImage2D(GL_TEXTURE_2D, 0, 4, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenBuffersARB(1, &vbo);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(float4) * width * height, NULL, GL_DYNAMIC_DRAW_ARB);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		cutilSafeCall(cudaGLRegisterBufferObject(vbo));

		initialized = true;
	}

	void release()
	{
		assert(initialized);

		cutilSafeCall(cudaGLUnregisterBufferObject(vbo));

		// delete vertex buffer
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		glDeleteBuffersARB(1, &vbo);

		// delete openGL texture
		glBindTexture(GL_TEXTURE_2D, 0);
		glDeleteTextures(1, &gltex);

		initialized = false;
	}

	float4 *map()
	{
		float4 *buf;
		cutilSafeCall(cudaGLMapBufferObject((void**)&buf, vbo));
		return buf;
	}

	void unmap()
	{
		cutilSafeCall(cudaGLUnmapBufferObject(vbo));
	}

	void draw(float x, float y, float screen_width, float screen_height)
	{
		// bind vertex buffer to texture
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
		glBindTexture(GL_TEXTURE_2D, gltex);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, NULL);

		// draw textured polygon
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(x, y);
		glTexCoord2f(1.0f, 0.0f); glVertex2f(x + screen_width, y);
		glTexCoord2f(1.0f, 1.0f); glVertex2f(x + screen_width, y + screen_height);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(x, y + screen_height);
		glEnd();

		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	}
};

#endif
