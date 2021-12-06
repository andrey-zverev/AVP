#pragma once

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <cstdint>
#include "helper_cuda.h"
#include "helper_image.h"
#include <math.h>

using namespace std;


unsigned char* GPU(unsigned char* img, int h, int w);
#include "MPIf.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <npp.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif
#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 8

__device__ __constant__ int kernelGPU[3][3] = {
	{ 0, 1, 0 },
	{ 1, 0, -1 },
	{ 0, -1, 0 },
};


__global__ void filter_kernel(unsigned char* inputBitmap, unsigned char* outputBitmap, int height, int dwordWidth, int width) {
	const int xIndex = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
	const int yIndex = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;


	if (xIndex >= dwordWidth || yIndex >= height)
		return;
	int threadAbsX = xIndex * 4;
	unsigned int result = 0;
	for (int k = 0; k < 4; k++)
	{
		int byteAbsX = threadAbsX + k;
		int offsetX, offsetY, absX, absY;
		int sum = 0;

		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				offsetX = (i - 1);
				offsetY = (j - 1);
				absX = byteAbsX + offsetX;
				absY = yIndex + offsetY;
				if (absX < 0 || absX >= width)
					absX = byteAbsX;
				if (absY < 0 || absY >= height)
					absY = yIndex;
				sum += inputBitmap[absX + absY * width] * kernelGPU[j][i];
			}
		}
		sum += 128;
		if (sum < 0) sum = 0;
		if (sum > 255) sum = 255;
		((unsigned char*)&result)[k] = sum;
	}
	((unsigned char*)outputBitmap)[xIndex + yIndex * dwordWidth] = result;
}

unsigned char* GPU(unsigned char* pixelData, int width, int height)
{
	size_t size = width * height;
	int start, finish, freq;
	unsigned char* pixelDataGPU, * resultGPU;
	size_t pitch;

	cudaMalloc((void**)&pixelDataGPU, size);
	cudaMalloc((void**)&resultGPU, size);
	cudaMemcpy(pixelDataGPU, pixelData, size, cudaMemcpyHostToDevice);

	int dwordWidth = (width + 3) / 4;
	int gridSize_X = (int)ceil((double)dwordWidth / (double)BLOCK_SIZE_X);
	int gridSize_Y = (int)ceil((double)height / (double)BLOCK_SIZE_Y);
	dim3 dimGrid(gridSize_X, gridSize_Y);
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	//filter_kernel <<<dimGrid, dimBlock>>> (pixelDataGPU, resultGPU, height, dwordWidth, width);
	cudaDeviceSynchronize();
	printf("\ntime CUDA = %lf\n", time);
	unsigned char* result = new unsigned char[size];
	cudaMemcpy(result, resultGPU, size, cudaMemcpyDeviceToHost);
	cudaFree(pixelDataGPU);
	cudaFree(resultGPU);
	cudaDeviceReset();
	return result;
}