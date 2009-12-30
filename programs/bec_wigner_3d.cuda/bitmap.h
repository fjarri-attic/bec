#ifndef _BITMAP_H
#define _BITMAP_H

#include <stdlib.h>
#include <stdio.h>

typedef unsigned long  uint32;
typedef unsigned short uint16;
typedef unsigned char  byte;

// Note: the magic number has been removed from the bmp_header structure
// since it causes alignment problems
//   bmpfile_magic_t should be written/read first
// followed by the
//   bmpfile_header_t
// [this avoids compiler-specific alignment pragmas etc.]

typedef struct {
  byte   magic[2];
} bmpfile_magic_t;

typedef struct {
  uint32 filesz;
  uint16 creator1;
  uint16 creator2;
  uint32 bmp_offset;
} bmpfile_header_t;

typedef struct {
  uint32 header_sz;
  uint32 width;
  uint32 height;
  uint16 nplanes;
  uint16 bitspp;
  uint32 compress_type;
  uint32 bmp_bytesz;
  uint32 hres;
  uint32 vres;
  uint32 ncolors;
  uint32 nimpcolors;
} bmp_dib_v3_header_t;

typedef enum {
  BI_RGB = 0,
  BI_RLE8,
  BI_RLE4,
  BI_BITFIELDS,
  BI_JPEG,
  BI_PNG,
} bmp_compression_method_t;

int getLineWidth(int width)
{
	return ((width * 3) % 4) ? (((width * 3) / 4 + 1) * 4) : (width * 3);
}

void writeHeaders(FILE *f, int width, int height)
{
	bmpfile_magic_t magic;
	bmpfile_header_t header;
	bmp_dib_v3_header_t bmpheader;
	int line_width = getLineWidth(width);

	magic.magic[0] = 0x42;
	magic.magic[1] = 0x4D;

	header.filesz = sizeof(header) + sizeof(magic) + sizeof(bmpheader) + line_width * height;
	header.creator1 = 0;
	header.creator2 = 0;
	header.bmp_offset = sizeof(header) + sizeof(magic) + sizeof(bmpheader);

	memset(&bmpheader, 0, sizeof(bmpheader));
	bmpheader.header_sz = sizeof(bmpheader);
	bmpheader.width = width;
	bmpheader.height = height;
	bmpheader.nplanes = 1;
	bmpheader.bitspp = 24;
	bmpheader.compress_type = BI_RGB;
	bmpheader.bmp_bytesz = line_width * height;
	bmpheader.hres = 300;
	bmpheader.vres = 300;
	bmpheader.ncolors = 0;
	bmpheader.nimpcolors = 0;

	fwrite(&magic, 1, sizeof(magic), f);
	fwrite(&header, 1, sizeof(header), f);
	fwrite(&bmpheader, 1, sizeof(bmpheader), f);
}

void createBitmap(char *file_name, int4 *rgba_data, int width, int height)
{
	int line_width = getLineWidth(width);

	FILE *f = fopen(file_name, "wb");
	writeHeaders(f, width, height);

	unsigned char *buffer = new unsigned char[line_width * height];
	memset(buffer, 0, line_width * height);
	for(int y = height - 1; y >= 0; y--)
		for(int x = 0; x < width; x++)
		{
			buffer[y * line_width + x * 3] = rgba_data[(height - y - 1) * width + x].z;
			buffer[y * line_width + x * 3 + 1] = rgba_data[(height - y - 1) * width + x].y;
			buffer[y * line_width + x * 3 + 2] = rgba_data[(height - y - 1) * width + x].x;
		}

	fwrite(buffer, 1, line_width * height, f);

	delete[] buffer;
	fclose(f);
}

unsigned char denormalize(float n)
{
	int res = (int)(n * 255);
	if(n < 0)
		return 0;
	if(n > 255)
		return 255;
	return (unsigned char)res;
}

void createBitmap(char *file_name, float4 *rgba_data, int width, int height)
{
	int line_width = getLineWidth(width);

	FILE *f = fopen(file_name, "wb");
	writeHeaders(f, width, height);

	unsigned char *buffer = new unsigned char[line_width * height];
	memset(buffer, 0, line_width * height);
	for(int y = height - 1; y >= 0; y--)
		for(int x = 0; x < width; x++)
		{
			buffer[y * line_width + x * 3] = denormalize(rgba_data[(height - y - 1) * width + x].z);
			buffer[y * line_width + x * 3 + 1] = denormalize(rgba_data[(height - y - 1) * width + x].y);
			buffer[y * line_width + x * 3 + 2] = denormalize(rgba_data[(height - y - 1) * width + x].x);
		}

	fwrite(buffer, 1, line_width * height, f);

	delete[] buffer;
	fclose(f);
}


#endif
