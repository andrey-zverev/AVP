#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <windows.h>
#include <conio.h>
#include <cstdint>
#include <chrono>
#include <npp.h>
#include "helper_cuda.h"
#include "helper_image.h"
//#include "Main.cpp"
using namespace std;

struct  Pixel {
	unsigned char red;
	unsigned char gren;
	unsigned char blue;
};

__device__ __constant__ int filterDevice[9];
const int BLOCKDIM_Y = 8;
const int BLOCKDIM_X = 32;
const int ELEMENT_IN_THREAD_WIDTH = 4;
const int BLOCK_ELEMENT_X = BLOCKDIM_X * ELEMENT_IN_THREAD_WIDTH;

__forceinline__ __device__ Pixel get_element(Pixel* array, unsigned int height, unsigned int width, unsigned int width_size, size_t pitch);
__global__ void GPUfunc(Pixel* img_origin, Pixel* img_new, int w, int h, int dif);
__global__ void GPUfuncShared(Pixel* img_origin, Pixel* img_new, int w, int h, int dif, size_t original_pitch, size_t result_pitch);
Pixel* GPUShared(Pixel* img, int h, int w);
Pixel* GPU(Pixel* img, int h, int w);
void compose(unsigned char* matrix1, unsigned char* matrix2, int h, int w);
inline unsigned int ind(unsigned int height, unsigned int width, unsigned int width_size);
Pixel* inStruct(unsigned char* img, int w, int h);
Pixel* rebuildPixel(Pixel* pixelStruct, int w, int h);
Pixel* CPU(Pixel* img, int h, int w);
Pixel filterCPU(Pixel* imgPixel, int i, int j, int w, int filter[3][3], int divisionCoef);
unsigned char* recoverImg(Pixel* structImg, int w, int h);

__global__ void GPUfunc(Pixel* img_origin, Pixel* img_new, int w, int h, int dif) {
	int curWidth = blockIdx.x * blockDim.x + threadIdx.x;
	int curHeight = blockIdx.y * blockDim.y + threadIdx.y;

	int filter[3][3] = {
		{ -1, -1, -1 },{ -1, 8,-1 },{ -1,-1, -1 }
	};

	//    0 1 0                  a b c
	//    1 0 -1                 d e f
							//	 z x n
	//    0 -1 0                {i * m + j]

	img_new[curWidth * w + curHeight].red = ((img_origin[curWidth * (w + 2) + curHeight].red * (filter[0][0])
		+ img_origin[(curWidth) * (w + 2) + (curHeight + 1)].red * (filter[0][1]) + img_origin[(curWidth) * (w + 2)
		+ (curHeight + 2)].red * (filter[0][2]) + img_origin[(curWidth + 1) * (w + 2) + (curHeight)].red * (filter[1][0])
		+ img_origin[(curWidth + 1) * (w + 2) + (curHeight + 1)].red * (filter[1][1]) + img_origin[(curWidth + 1) * (w + 2)
		+ (curHeight + 2)].red * (filter[1][2]) + img_origin[(curWidth + 2) * (w + 2) + (curHeight)].red * (filter[2][0])
		+ img_origin[(curWidth + 2) * (w + 2) + (curHeight + 1)].red * (filter[2][1]) + img_origin[(curWidth + 2) * (w + 2)
		+ (curHeight + 2)].red * (filter[2][2])) / dif);

	img_new[curWidth * w + curHeight].gren = ((img_origin[curWidth * (w + 2) + curHeight].gren * (filter[0][0])
		+ img_origin[(curWidth) * (w + 2) + (curHeight + 1)].gren * (filter[0][1]) + img_origin[(curWidth) * (w + 2)
		+ (curHeight + 2)].gren * (filter[0][2]) + img_origin[(curWidth + 1) * (w + 2) + (curHeight)].gren * (filter[1][0])
		+ img_origin[(curWidth + 1) * (w + 2) + (curHeight + 1)].gren * (filter[1][1]) + img_origin[(curWidth + 1) * (w + 2)
		+ (curHeight + 2)].gren * (filter[1][2]) + img_origin[(curWidth + 2) * (w + 2) + (curHeight)].gren * (filter[2][0])
		+ img_origin[(curWidth + 2) * (w + 2) + (curHeight + 1)].gren * (filter[2][1]) + img_origin[(curWidth + 2) * (w + 2)
		+ (curHeight + 2)].gren * (filter[2][2])) / dif);

	img_new[curWidth * w + curHeight].blue = ((img_origin[curWidth * (w + 2) + curHeight].blue * (filter[0][0])
		+ img_origin[(curWidth) * (w + 2) + (curHeight + 1)].blue * (filter[0][1]) + img_origin[(curWidth) * (w + 2)
		+ (curHeight + 2)].blue * (filter[0][2]) + img_origin[(curWidth + 1) * (w + 2) + (curHeight)].blue * (filter[1][0])
		+ img_origin[(curWidth + 1) * (w + 2) + (curHeight + 1)].blue * (filter[1][1]) + img_origin[(curWidth + 1) * (w + 2)
		+ (curHeight + 2)].blue * (filter[1][2]) + img_origin[(curWidth + 2) * (w + 2) + (curHeight)].blue * (filter[2][0])
		+ img_origin[(curWidth + 2) * (w + 2) + (curHeight + 1)].blue * (filter[2][1]) + img_origin[(curWidth + 2) * (w + 2)
		+ (curHeight + 2)].blue * (filter[2][2])) / dif);
}

__global__ void GPUfuncShared(Pixel* img_origin, Pixel* img_new, int w, int h, int dif, size_t original_pitch, size_t result_pitch) {

	int result_current_width = (blockIdx.x * BLOCK_ELEMENT_X) + (threadIdx.x * ELEMENT_IN_THREAD_WIDTH);
	int result_current_height = (blockDim.y * blockIdx.y) + threadIdx.y;

	int original_current_width = result_current_width + 1;
	int original_current_height = result_current_height + 1;


	__shared__ Pixel temp_image[BLOCKDIM_Y + 2][BLOCK_ELEMENT_X + 2];

	{
		temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] =
			get_element(
				img_origin,
				original_current_height,
				original_current_width,
				w + 2,
				original_pitch
			);

		temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] =
			get_element(
				img_origin,
				original_current_height,
				original_current_width + 1,
				w + 2,
				original_pitch
			);

		temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3] =
			get_element(
				img_origin,
				original_current_height,
				original_current_width + 2,
				w + 2,
				original_pitch
			);

		temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 4] =
			get_element(
				img_origin,
				original_current_height,
				original_current_width + 3,
				w + 2,
				original_pitch
			);
	}
	{
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			temp_image[0][0] =
				get_element(
					img_origin,
					original_current_height - 1,
					original_current_width - 1,
					w,
					original_pitch
				);
		}

		if (threadIdx.x == BLOCKDIM_X - 1 && threadIdx.y == 0)
		{
			temp_image[0][BLOCK_ELEMENT_X + 1] =
				get_element(
					img_origin,
					original_current_height - 1,
					original_current_width + 4,
					w + 2,
					original_pitch
				);
		}

		if (threadIdx.x == BLOCKDIM_X - 1 && threadIdx.y == BLOCKDIM_Y - 1)
		{
			temp_image[BLOCKDIM_Y + 1][BLOCK_ELEMENT_X + 1] =
				get_element(
					img_origin,
					original_current_height + 1,
					original_current_width + 4,
					w + 2,
					original_pitch
				);
		}

		if (threadIdx.x == 0 && threadIdx.y == BLOCKDIM_Y - 1)
		{
			temp_image[BLOCKDIM_Y + 1][0] =
				get_element(
					img_origin,
					original_current_height + 1,
					original_current_width - 1,
					w + 2,
					original_pitch
				);
		}
	}
	{
		if (threadIdx.x == 0)
		{
			temp_image[threadIdx.y + 1][0] =
				get_element(
					img_origin,
					original_current_height,
					original_current_width - 1,
					w,
					original_pitch
				);
		}

		if (threadIdx.x == BLOCKDIM_X - 1)
		{
			temp_image[threadIdx.y + 1][BLOCK_ELEMENT_X + 1] =
				get_element(
					img_origin,
					original_current_height,
					original_current_width + 4,
					w + 2,
					original_pitch
				);
		}

		if (threadIdx.y == 0)
		{
			temp_image[0][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] =
				get_element(
					img_origin,
					original_current_height - 1,
					original_current_width,
					w + 2,
					original_pitch
				);

			temp_image[0][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] =
				get_element(
					img_origin,
					original_current_height - 1,
					original_current_width + 1,
					w + 2,
					original_pitch
				);

			temp_image[0][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3] =
				get_element(
					img_origin,
					original_current_height - 1,
					original_current_width + 2,
					w + 2,
					original_pitch
				);

			temp_image[0][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 4] =
				get_element(
					img_origin,
					original_current_height - 1,
					original_current_width + 3,
					w + 2,
					original_pitch
				);
		}
		if (threadIdx.y == BLOCKDIM_Y - 1)
		{
			temp_image[BLOCKDIM_Y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] =
				get_element(
					img_origin,
					original_current_height + 1,
					original_current_width,
					w + 2,
					original_pitch
				);

			temp_image[BLOCKDIM_Y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] =
				get_element(
					img_origin,
					original_current_height + 1,
					original_current_width + 1,
					w + 2,
					original_pitch
				);

			temp_image[BLOCKDIM_Y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3] =
				get_element(
					img_origin,
					original_current_height + 1,
					original_current_width + 2,
					w + 2,
					original_pitch
				);

			temp_image[BLOCKDIM_Y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 4] =
				get_element(
					img_origin,
					original_current_height + 1,
					original_current_width + 3,
					w + 2,
					original_pitch
				);
		}
	}

	__syncthreads();

	Pixel* elemet = (Pixel*)((unsigned char*)img_new + result_current_height * result_pitch);
	{
		elemet[result_current_width].red = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].red * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].red * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].red * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].red * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].red * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].red * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].red * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].red * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].red * (filterDevice[8])
				)
			/ dif
			);

		elemet[result_current_width].gren = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].gren * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].gren * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].gren * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].gren * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].gren * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].gren * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].gren * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].gren * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].gren * (filterDevice[8])
				)
			/ dif
			);

		elemet[result_current_width].blue = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].blue * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].blue * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].blue * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].blue * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].blue * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].blue * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)].blue * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].blue * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].blue * (filterDevice[8])
				)
			/ dif
			);
	}
	{
		elemet[result_current_width + 1].red = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].red * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].red * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].red * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].red * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].red * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].red * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].red * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].red * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].red * (filterDevice[8])
				)
			/ dif
			);

		elemet[result_current_width + 1].gren = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].gren * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].gren * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].gren * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].gren * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].gren * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].gren * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].gren * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].gren * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].gren * (filterDevice[8])
				)
			/ dif
			);

		elemet[result_current_width + 1].blue = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].blue * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].blue * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].blue * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].blue * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].blue * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].blue * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1].blue * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1].blue * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1].blue * (filterDevice[8])
				)
			/ dif
			);
	}
	{
		elemet[result_current_width + 2].red = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].red * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].red * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].red * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].red * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].red * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].red * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].red * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].red * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].red * (filterDevice[8])
				)
			/ dif
			);

		elemet[result_current_width + 2].gren = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].gren * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].gren * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].gren * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].gren * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].gren * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].gren * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].gren * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].gren * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].gren * (filterDevice[8])
				)
			/ dif
			);

		elemet[result_current_width + 2].blue = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].blue * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].blue * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].blue * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].blue * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].blue * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].blue * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2].blue * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2].blue * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2].blue * (filterDevice[8])
				)
			/ dif
			);
	}
	{
		elemet[result_current_width + 3].red = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].red * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].red * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].red * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].red * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].red * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].red * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].red * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].red * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].red * (filterDevice[8])
				)
			/ dif
			);

		elemet[result_current_width + 3].gren = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].gren * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].gren * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].gren * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].gren * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].gren * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].gren * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].gren * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].gren * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].gren * (filterDevice[8])
				)
			/ dif
			);

		elemet[result_current_width + 3].blue = (
			(
				temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].blue * (filterDevice[0])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].blue * (filterDevice[1])
				+ temp_image[threadIdx.y][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].blue * (filterDevice[2])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].blue * (filterDevice[3])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].blue * (filterDevice[4])
				+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].blue * (filterDevice[5])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3].blue * (filterDevice[6])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3].blue * (filterDevice[7])
				+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3].blue * (filterDevice[8])
				)
			/ dif
			);
	}
}

Pixel* GPUShared(Pixel* img, int h, int w) {
	int dif = 1;

	int filterHost[9] =
	{ -1, -1, -1 , -1,8,-1 , -1,-1, -1 };

	cudaMemcpyToSymbol(filterDevice, filterHost, sizeof(int) * 9);

	cudaError_t cudaStatus;

	cudaEvent_t startTime;
	cudaEvent_t stopTime;

	size_t image_original_pitch;
	size_t image_result_pitch;

	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);

	float resultTime;

	Pixel* result = new Pixel[(w) * (h)];
	Pixel* img_origin;
	Pixel* img_new;

	cudaStatus = cudaMallocPitch((void**)(&img_origin), &image_original_pitch, (w + 2) * sizeof(Pixel), h + 2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!1");
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpy2D(img_origin, image_original_pitch, img, (w + 2) * sizeof(Pixel), (w + 2) * sizeof(Pixel), h + 2, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!2");
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMallocPitch((void**)(&img_new), &image_result_pitch, w * sizeof(Pixel), h);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!3");
		exit(EXIT_FAILURE);
	}

	dim3 bl(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid;

	grid.x = w / BLOCK_ELEMENT_X;
	if (w % BLOCK_ELEMENT_X != 0)
		grid.x += 1;

	grid.y = h / bl.y;
	if (h % BLOCKDIM_Y != 0)
		grid.y += 1;

	cudaEventRecord(startTime, 0);

	GPUfuncShared << <grid, bl >> > (img_origin, img_new, w, h, dif, image_original_pitch, image_result_pitch);

	cudaEventRecord(stopTime, 0);
	cudaEventSynchronize(stopTime);
	cudaEventElapsedTime(&resultTime, startTime, stopTime);

	cout << "GPU Shared time:" << resultTime << " ms" << endl;


	cudaStatus = cudaMemcpy2D(result, w * sizeof(Pixel), img_new, image_result_pitch, w * sizeof(Pixel), h, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!(4)\n");
		exit(EXIT_FAILURE);
	}

	cudaThreadSynchronize();

	return result;
}

Pixel* GPU(Pixel* img, int h, int w) {
	int dif = 1;

	cudaError_t cudaStatus;

	cudaEvent_t startTime;
	cudaEvent_t stopTime;

	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);

	float resultTime;

	Pixel* result = new Pixel[h * w];
	Pixel* img_origin;

	cudaStatus = cudaMalloc((void**)&img_origin, (h + 2) * (w + 2) * sizeof(Pixel));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!(1)");
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpy(img_origin, img, (h + 2) * (w + 2) * sizeof(Pixel), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!(1)");
		exit(EXIT_FAILURE);
	}

	Pixel* img_new;
	cudaStatus = cudaMalloc((void**)&img_new, (h) * (w) * sizeof(Pixel));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!(2)");
		exit(EXIT_FAILURE);
	}

	dim3 bl(8, 32);
	dim3 grid;

	grid.x = h / bl.x;
	if (h % bl.x != 0)
		grid.x += 1;

	grid.y = w / bl.y;
	if (w % bl.y != 0)
		grid.y += 1;

	cudaEventRecord(startTime);
	GPUfunc << <grid, bl >> > (img_origin, img_new, w, h, dif);
	cudaDeviceSynchronize();

	cudaEventRecord(stopTime);
	cudaEventSynchronize(stopTime);
	cudaThreadSynchronize();

	cudaEventElapsedTime(&resultTime, startTime, stopTime);

	cout << "GPU time:" << resultTime << " ms" << endl;
	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(result, img_new, h * w * sizeof(Pixel), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!(2)");
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();

	cudaFree(img_new);
	cudaFree(img_origin);

	return result;
}

__forceinline__ __device__ Pixel get_element(Pixel* array, unsigned int height, unsigned int width, unsigned int width_size, size_t pitch) {
	Pixel* elemet = (Pixel*)((unsigned char*)array + height * pitch);
	return elemet[width];
}

int main() {
	unsigned char* img = NULL;
	unsigned char* rebuild = NULL;
	unsigned int w = 0;
	unsigned int h = 0;
	unsigned int chanels = 3;

	__loadPPM("../images/Hero2.ppm", &img, &w, &h, &chanels);

	Pixel* structPixel = inStruct(img, w, h);

	Pixel* structPixelRebuild = rebuildPixel(structPixel, w, h);

	Pixel* structPixelCPU = CPU(structPixelRebuild, h, w);
	Pixel* structPixelGPU = GPU(structPixelRebuild, h, w);
	Pixel* structPixelGPUShared = GPUShared(structPixelRebuild, h, w);

	unsigned char* imgCPU = recoverImg(structPixelCPU, w, h);
	unsigned char* imgGPU = recoverImg(structPixelGPU, w, h);
	unsigned char* imgGPUShared = recoverImg(structPixelGPUShared, w, h);

	compose(imgCPU, imgGPU, h, w);
	compose(imgCPU, imgGPUShared, h, w);

	cout << "Image size: " << h << "x" << w << endl;

	__savePPM("../images/CPU.ppm", imgCPU, w, h, chanels);
	__savePPM("../images/GPU.ppm", imgGPU, w, h, chanels);
	__savePPM("../images/GPUShared.ppm", imgGPUShared, w, h, chanels);

	system("pause");
}

Pixel* inStruct(unsigned char* img, int w, int h) {
	int fullWidth = w * 3;
	int fullHeight = h * 3;

	Pixel* pixelStruct = new Pixel[w * h];

	int i = 0;
	int j = 0;

	while (i < h * w) {
		pixelStruct[i].red = img[j];
		pixelStruct[i].gren = img[j + 1];
		pixelStruct[i].blue = img[j + 2];

		i++;
		j += 3;
	}

	return pixelStruct;
}

Pixel* rebuildPixel(Pixel* pixelStruct, int w, int h) {
	int newW = w + 2;
	int newH = h + 2;

	Pixel* newPixelStruct = new Pixel[(w + 2) * (h + 2)];

	newPixelStruct[0] = pixelStruct[0];
	newPixelStruct[ind(0, newW - 1, newW)] = pixelStruct[ind(0, w - 1, w)];
	newPixelStruct[ind(newH - 1, 0, newW)] = pixelStruct[ind(h - 1, 0, w)];
	newPixelStruct[ind(newH - 1, newW - 1, newW)] = pixelStruct[ind(h - 1, w - 1, w)];

	for (int i = 0; i < h; i++)
	{
		newPixelStruct[ind(i + 1, 0, newW)] = pixelStruct[ind(i, 0, w)];
		newPixelStruct[ind(i + 1, newW - 1, newW)] = pixelStruct[ind(i, w - 1, w)];
	}

	for (int j = 0; j < w; j++)
	{
		newPixelStruct[ind(0, j + 1, newW)] = pixelStruct[ind(0, j, w)];
		newPixelStruct[ind(newH - 1, j + 1, newW)] = pixelStruct[ind(h - 1, j, w)];
	}

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			newPixelStruct[ind(i + 1, j + 1, newW)] = pixelStruct[ind(i, j, w)];
		}
	}

	return newPixelStruct;
}

inline unsigned int ind(unsigned int height, unsigned int width, unsigned int width_size) {
	return height * width_size + width;
}

Pixel* CPU(Pixel* img, int h, int w) {
	int filter[3][3] = {
		{ -1, -1, -1 },{ -1, 8,-1 },{ -1,-1, -1 }
	};
	int divisionCoef = 1;

	Pixel* newImg = new Pixel[h * w];

	chrono::time_point<chrono::steady_clock> start, end;
	start = chrono::steady_clock::now();
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			newImg[(i)*w + (j)] = filterCPU(img, i, j, w, filter, divisionCoef);
		}
	}
	end = chrono::steady_clock::now();
	auto CPU_TIME = end - start;
	cout << "CPU time:" << chrono::duration <double, milli>(CPU_TIME).count() << " ms" << endl;
	return newImg;
}

Pixel filterCPU(Pixel* imgPixel, int i, int j, int w, int filter[3][3], int divisionCoef) {
	Pixel pixel;

	pixel.red = ((imgPixel[i * (w + 2) + j].red * (filter[0][0]) + imgPixel[(i) * (w + 2)
		+ (j + 1)].red * (filter[0][1]) + imgPixel[(i) * (w + 2) + (j + 2)].red * (filter[0][2])
		+ imgPixel[(i + 1) * (w + 2) + (j)].red * (filter[1][0]) + imgPixel[(i + 1) * (w + 2) + (j + 1)].red * (filter[1][1])
		+ imgPixel[(i + 1) * (w + 2) + (j + 2)].red * (filter[1][2]) + imgPixel[(i + 2) * (w + 2) + (j)].red * (filter[2][0])
		+ imgPixel[(i + 2) * (w + 2) + (j + 1)].red * (filter[2][1]) + imgPixel[(i + 2) * (w + 2) + (j + 2)].red * (filter[2][2])) / divisionCoef);

	pixel.gren = ((imgPixel[i * (w + 2) + j].gren * (filter[0][0]) + imgPixel[(i) * (w + 2)
		+ (j + 1)].gren * (filter[0][1]) + imgPixel[(i) * (w + 2) + (j + 2)].gren * (filter[0][2])
		+ imgPixel[(i + 1) * (w + 2) + (j)].gren * (filter[1][0]) + imgPixel[(i + 1) * (w + 2) + (j + 1)].gren * (filter[1][1])
		+ imgPixel[(i + 1) * (w + 2) + (j + 2)].gren * (filter[1][2]) + imgPixel[(i + 2) * (w + 2) + (j)].gren * (filter[2][0])
		+ imgPixel[(i + 2) * (w + 2) + (j + 1)].gren * (filter[2][1]) + imgPixel[(i + 2) * (w + 2) + (j + 2)].gren * (filter[2][2])) / divisionCoef);

	pixel.blue = ((imgPixel[i * (w + 2) + j].blue * (filter[0][0]) + imgPixel[(i) * (w + 2)
		+ (j + 1)].blue * (filter[0][1]) + imgPixel[(i) * (w + 2) + (j + 2)].blue * (filter[0][2])
		+ imgPixel[(i + 1) * (w + 2) + (j)].blue * (filter[1][0]) + imgPixel[(i + 1) * (w + 2) + (j + 1)].blue * (filter[1][1])
		+ imgPixel[(i + 1) * (w + 2) + (j + 2)].blue * (filter[1][2]) + imgPixel[(i + 2) * (w + 2) + (j)].blue * (filter[2][0])
		+ imgPixel[(i + 2) * (w + 2) + (j + 1)].blue * (filter[2][1]) + imgPixel[(i + 2) * (w + 2) + (j + 2)].blue * (filter[2][2])) / divisionCoef);

	return pixel;
}

unsigned char* recoverImg(Pixel* structImg, int w, int h) {
	int fullWidth = w * 3;
	int fullHeight = h * 3;

	unsigned char* img = new unsigned char[(w * 3) * (3 * h)];

	int i = 0;
	int j = 0;

	while (j < w * h) {
		img[i] = structImg[j].red;
		img[i + 1] = structImg[j].gren;
		img[i + 2] = structImg[j].blue;

		i += 3;
		j += 1;
	}

	return img;
}

void compose(unsigned char* matrix1, unsigned char* matrix2, int h, int w) {
	int t = 0;
	for (int i = 0; i < h * w; i++)
		if (matrix1[i] != matrix2[i]) {
			t++;
		}
	cout << "Count of miss: " << t << endl;
}