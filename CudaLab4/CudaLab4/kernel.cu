#include <cstdlib>
#include <ostream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <windows.h>

#define Mbig 27700
#define Nbig 12300
#define BLOCK_SIZE 16
#define M 2570
#define N 2270

using namespace std;

void checkOnError(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA return error code: " << cudaStatus;
		cout << " " << cudaGetErrorString(cudaStatus) << endl;
	}
}

void randomElements(short* matrix, long long matrixRows, long long matrixCols) {
	srand(time(NULL));
	for (long long i = 0; i < matrixRows; i++) {
		for (long long j = 0; j < matrixCols; j++) 
			matrix[i * matrixCols + j] = rand() % 100 + 1;
	}
}

void showMatrix(short* matrix, int matrixRows, int matrixCols) {
	for (int i = 0; i < matrixRows; i++) {
		for (int j = 0; j < matrixCols; j++)
			cout << setw(4) << matrix[i * matrixCols + j];
		cout << '\n';
	}
}

__global__ void kernel(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, short* resultMatrix) {
	int rows = blockIdx.y * blockDim.y + threadIdx.y;
	int cols = blockIdx.x * blockDim.x + threadIdx.x;
	if ((rows <= sourseMatrixRow) && (cols <= sourseMatrixCol)) {
		resultMatrix[(rows * sourseMatrixCol + cols) * 2] = sourseMatrix[rows * sourseMatrixCol + cols];
		__syncthreads();
		resultMatrix[(rows * sourseMatrixCol + cols) * 2 + 1] = sourseMatrix[rows * sourseMatrixCol + cols];
		__syncthreads();
	}
}

void transformMatrixGPU(short* matrix_In, int rows, int cols, short* result) {
	cudaEvent_t start;
	cudaEvent_t stop;
	short* matIn;
	short* matOut;
	float time;
	checkOnError(cudaMalloc((void**)&matIn, rows * cols * sizeof(short)));
	checkOnError(cudaMemcpy(matIn, matrix_In, rows * cols * sizeof(short), cudaMemcpyHostToDevice));
	checkOnError(cudaMalloc((void**)&matOut, rows * cols * 2 * sizeof(short)));
	dim3 block(16, 4);
	dim3 grid(cols / block.x, rows / block.y);
	if (cols % block.x != 0) grid.x++;
	if (rows % block.y != 0) grid.y++;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	kernel << <grid, block >> > (
		matIn, 
		rows,
		cols, 
		matOut);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "GPU time: " << time << " ms" << endl;
	cudaMemcpy(result, matOut, rows * cols * 2 * sizeof(short),
		cudaMemcpyDeviceToHost);
	checkOnError(cudaFree(matIn));
	checkOnError(cudaFree(matOut));
}

bool compareMatrix(short* matrix1, short* matrix2, int matrixRow, int matrixCol) {
	for (auto i = 0; i < matrixRow; i++)
		for (auto j = 0; j < matrixCol; j++)
			if (matrix1[i * matrixCol + j] != matrix2[i * matrixCol + j])
				return true;
	return false;
}

__global__ void kernelSharedGpu(short* first_matrix, int first_matrix_height, int first_matrix_width, short* second_matrix, const int second_matrix_height, const int second_matrix_width) {
	int xIndex = blockIdx.x * 8 + threadIdx.x;
	int yIndex = blockIdx.y * 8 + threadIdx.y;
	int idx = yIndex * first_matrix_width + xIndex;
	int index_out = (yIndex * first_matrix_width + xIndex) * 2;
	__shared__ short block[8][8 * 2];
	if ((xIndex <= first_matrix_width) && (yIndex <= first_matrix_height))
	{
		block[threadIdx.y][threadIdx.x * 2] = first_matrix[idx]; 
		__syncthreads();
		second_matrix[index_out] = block[threadIdx.y][threadIdx.x * 2];
		second_matrix[index_out + 1] = block[threadIdx.y][threadIdx.x * 2];
	}
}

float transform_matrix_gpu_shared(short* first_matrix, const int first_matrix_height, const int first_matrix_width, short* second_matrix, const int second_matrix_height, const int second_matrix_width) {
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	short* gpu_first_matrix;
	short* gpu_second_matrix;
	size_t pitch;

	checkOnError(cudaMalloc((void**)&gpu_first_matrix, M*N*sizeof(short)));
	checkOnError(cudaMemcpy(gpu_first_matrix, first_matrix, M * N * sizeof(short), cudaMemcpyHostToDevice));
	checkOnError(cudaMalloc((void**)&gpu_second_matrix, M * 2 * N * sizeof(short)));

	dim3 block(8, 8);
	dim3 grid;

	grid.x = first_matrix_width / block.x;
	if (first_matrix_width % block.x != 0) grid.x += 1;

	grid.y = first_matrix_height / block.y;
	if (first_matrix_height % block.y != 0) grid.y += 1;

	checkOnError(cudaEventCreate(&startTime));
	checkOnError(cudaEventCreate(&stopTime));
	checkOnError(cudaEventRecord(startTime));

	kernelSharedGpu << <grid, block >> > (
		gpu_first_matrix,	
		first_matrix_height, 
		first_matrix_width, 
		gpu_second_matrix,
		second_matrix_height,
		second_matrix_width);

	checkOnError(cudaEventRecord(stopTime));
	checkOnError(cudaEventSynchronize(stopTime));
	float result_time;
	checkOnError(cudaEventElapsedTime(&result_time, startTime, stopTime));
	cout << "Shared GPU time: " << result_time << " ms" << endl;
	cudaMemcpy(second_matrix, gpu_second_matrix,
		second_matrix_height * second_matrix_width * sizeof(short),
		cudaMemcpyDeviceToHost);

	return result_time;
	checkOnError(cudaFree(gpu_first_matrix));
	checkOnError(cudaFree(gpu_second_matrix));
}

void transformBigMatrixGPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, short* resultMatrix) {
	int PARTS = M / 2;
	if (M % 2 != 0) PARTS++;
	short** arrayOfMatrices;
	arrayOfMatrices = (short**)malloc(PARTS * sizeof(short*));
	int i = 0;
	for (int k = 0; k < PARTS; k++)
	{
		arrayOfMatrices[k] = (short*)malloc(sourseMatrixCol * 2 * sizeof(short));
		for (int j = 0; j < N; j++)
		{
		arrayOfMatrices[k][0 * sourseMatrixCol + j] = sourseMatrix[i * sourseMatrixCol + j];
		arrayOfMatrices[k][1 * sourseMatrixCol + j] = sourseMatrix[(i + 1) * sourseMatrixCol + j];
		}
		i += 2;
	}
	cout << '\n';
	short** resultmatrix;
	resultmatrix = (short**)malloc(PARTS * sizeof(short*));
	short* sourseMatrixGPU;
	short* resultMatrixGPU;
	float timeCounter;
	float resultTime = 0;
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	for (int partsCounter = 0; partsCounter < PARTS; partsCounter++) {
		checkOnError(cudaMalloc((void**)&sourseMatrixGPU, 2 * sourseMatrixCol * sizeof(short)));
		checkOnError(cudaMalloc((void**)&resultMatrixGPU, 2 * sourseMatrixCol * 2 * sizeof(short)));
		checkOnError(cudaMemcpy(sourseMatrixGPU, arrayOfMatrices[partsCounter], 2 * sourseMatrixCol * sizeof(short), cudaMemcpyHostToDevice));
		dim3 block(16, 8);
		dim3 grid(sourseMatrixCol / block.x, sourseMatrixRow / block.y);
		if (sourseMatrixRow % block.y != 0) 
			grid.y++;
		if (sourseMatrixCol % block.x != 0) 
			grid.x++;
		cudaEventCreate(&startTime);
		cudaEventCreate(&stopTime);
		cudaEventRecord(startTime);
		kernel << <grid, block >> > (sourseMatrixGPU, 2, sourseMatrixCol, resultMatrixGPU);
		cudaEventRecord(stopTime);
		cudaEventSynchronize(stopTime);
		cudaEventElapsedTime(&timeCounter, startTime, stopTime);
		resultTime += timeCounter;
		resultmatrix[partsCounter] = (short*)malloc(2 * 2 * sourseMatrixCol * sizeof(short));
		checkOnError(cudaMemcpy(resultmatrix[partsCounter], resultMatrixGPU, 2 * 2 * sourseMatrixCol * sizeof(short),
			cudaMemcpyDeviceToHost));
		checkOnError(cudaFree(sourseMatrixGPU));
		checkOnError(cudaFree(resultMatrixGPU));
	}
	cout << "GPU time: " << resultTime << " ms" << endl;
	i = 0;
	for (int k = 0; k < PARTS; k++)
	{

		for (int j = 0; j < N * 2; j++)
		{
			resultMatrix[i * sourseMatrixCol *2 + j]  = resultmatrix[k][0 * sourseMatrixCol * 2 + j];
			resultMatrix[(i + 1) * sourseMatrixCol*2 + j] = resultmatrix[k][1 * sourseMatrixCol * 2 + j];
		}
		i += 2;
	}
}


int main(int argc, char *argv[])
{
#pragma region init
	short* matrix_In;
	short* matrix_In_gpu;
	short* matrix_Out_Cpu;
	short* matrix_Out_Gpu;
	short* matrix_Out_Gpu_big;
	short* matrix_Out_Gpu_shared;
	matrix_In = (short*)malloc(M * N * sizeof(short)+1);
	randomElements(matrix_In, M, N);
	matrix_Out_Cpu = (short*)malloc(M * N * 2 * sizeof(short));
	matrix_Out_Gpu = (short*)malloc(M * N * 2 * sizeof(short));
	matrix_Out_Gpu_shared = (short*)malloc(M * N * 2 * sizeof(short));
	matrix_Out_Gpu_big = (short*)malloc(M * N * 2 * sizeof(short));
	int k = 0;
	auto start_cpu = chrono::steady_clock::now();
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			matrix_Out_Cpu[(i * N + j) * 2] = matrix_In[i * N + j];
			matrix_Out_Cpu[(i * N + j) * 2 + 1] = matrix_In[i * N + j];
		}
	}
	auto end_cpu = chrono::steady_clock::now();
	cout << "CPU time: " << chrono::duration <double, milli>(end_cpu - start_cpu).count() << " ms" << endl;
#pragma endregion

	transformMatrixGPU(matrix_In, M, N, matrix_Out_Gpu);
	transform_matrix_gpu_shared(matrix_In, M, N, matrix_Out_Gpu_shared, M, N * 2);
	//transformBigMatrixGPU(matrix_In, M, N, matrix_Out_Gpu_big);

#pragma region output
	cout << "Matrix in\n";
	showMatrix(matrix_In, 10, 10);
	cout << '\n';
	cout << '\n';
	cout << "Matrix out cpu\n";
	showMatrix(matrix_Out_Cpu, 10, 20);
	cout << '\n';
	cout << '\n';
	cout << "Matrix out gpu simple\n";
	showMatrix(matrix_Out_Gpu, 10, 20);
	cout << '\n';
	cout << '\n';
	cout << "Matrix out gpu shared\n";
	showMatrix(matrix_Out_Gpu_shared, 10, 20);
	cout << '\n';
	cout << '\n';
	cout << "Matrix out gpu big\n";
	showMatrix(matrix_Out_Gpu_big, 10, 20);
	if (!compareMatrix(matrix_Out_Cpu, matrix_Out_Gpu, M, N * 2))
		cout << "Matrix cpu and gpu are the same" << endl; else cout << "MATRIX CPU AND GPU NOT EQUAL" << endl;
	if (!compareMatrix(matrix_Out_Cpu, matrix_Out_Gpu_shared, M, N * 2))
		cout << "Matrix gpu and gpu_shared are the same" << endl; else cout << "MATRIX CPU AND GPU SHARED NOT EQUAL" << endl;
	if (!compareMatrix(matrix_Out_Cpu, matrix_Out_Gpu_big, M, N * 2))
		cout << "Matrix gpu and gpu_big are the same" << endl; else cout << "MATRIX CPU AND GPU BIG NOT EQUAL" << endl;
	checkOnError(cudaDeviceReset());
#pragma endregion
	system("pause");
    return 0;
}