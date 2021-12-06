#include "mpi.h"
//#include "MPIf.cuh"
//#include "MPIf.cu"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <cstdint>
#include "helper_cuda.h"
#include "helper_image.h"
#include <math.h>
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

using namespace std;

void compose(unsigned char* matrix1, unsigned char* matrix2, int h, int w);
inline unsigned int ind(unsigned int height, unsigned int width, unsigned int width_size);
int* generate(int world_size, int n);

unsigned char* rebuildSrcData(unsigned char* srcData, int w, int h) {//создание границ
	unsigned char* rebuildData = new unsigned char[(w + 2) * (h + 2)];//создаём новую матрицу пикселей
	int newW = w + 2;//новая ширина(кол-во столбиков)
	int newH = h + 2;//новая высота(кол-во строчек)
	//границы будем заполнять нулями т.к. МАХ - фильтор

	//4 строки заполнения углов
	rebuildData[0] = srcData[0];
	rebuildData[ind(0, newW - 1, newW)] = srcData[ind(0, w - 1, w)];
	rebuildData[ind(newH - 1, 0, newW)] = srcData[ind(h - 1, 0, w)];
	rebuildData[ind(newH - 1, newW - 1, newW)] = srcData[ind(h - 1, w - 1, w)];

	for (int i = 0; i < h; i++)//заполняем левый и правый столбик
	{
		rebuildData[ind(i + 1, 0, newW)] = srcData[ind(i, 0, w)];
		rebuildData[ind(i + 1, newW - 1, newW)] = srcData[ind(i, w - 1, w)];
	}

	for (int j = 0; j < w; j++)//верхняя и нижняя строка
	{
		rebuildData[ind(0, j + 1, newW)] = srcData[ind(0, j, w)];
		rebuildData[ind(newH - 1, j + 1, newW)] = srcData[ind(h - 1, j, w)];
	}

	for (int i = 0; i < h; i++) {//переносим остальные данный из изначальной матрицы
		for (int j = 0; j < w; j++) {
			rebuildData[ind(i + 1, j + 1, newW)] = srcData[ind(i, j, w)];
		}
	}

	return rebuildData;
}//создание границ

unsigned char filter(unsigned char* pixelData, int i, int j, int w)
{
	unsigned char result;

	result = pixelData[(i - 1) * w + j] + pixelData[i * w + (j - 1)] - pixelData[i * w + (j + 1)] - pixelData[(i + 1) * w + j] + 128;

	if (result < 0) result = 0;
	if (result > 255) result = 255;

	return result;
}

unsigned char* returnData(unsigned char* Data, int width, int height)
{
	int w = width - 2;//уменьшаем границы
	int h = height - 2;
	unsigned char* rData = new unsigned char[(w * h)];//новый буфер для картинки выходной
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			rData[i * w + j] = Data[(i + 1) * width + (j + 1)];//переносим значения ВНУТРИ границ, не включая границы
	return rData;
}

__forceinline__ __device__ unsigned char filterCUDA(unsigned char* pixelData, int i, int j, int w)
{
	unsigned char result;

	result = pixelData[(i - 1) * w + j] + pixelData[i * w + (j - 1)] - pixelData[i * w + (j + 1)] - pixelData[(i + 1) * w + j] + 128;

	if (result < 0) result = 0;
	if (result > 255) result = 255;

	return result;
}

__global__ void filter_kernel(unsigned char* inputBitmap, unsigned char* outputBitmap, int height, int width) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if (((xIndex > 0) && (yIndex > 0))&&((xIndex<height)&&(yIndex<width)))
	{
		//тупо берём и накладываем фильтр
		outputBitmap[xIndex * width + yIndex] = filterCUDA(inputBitmap, xIndex, yIndex, width);
		__syncthreads();
	}
}

unsigned char* filter_CPU(unsigned char* pixelData, int w, int h) {
	unsigned char* result = new unsigned char[w * h];//создаём результирующую матрицу
	if (result == NULL)
		return NULL;
	//тупо поэлементно накладываем фильтр
	for (int i = 1; i < h - 1; i++) {
		for (int j = 1; j < w - 1; j++) {
			result[(i)*w + (j)] = filter(pixelData, i, j, w);
		}
	}
	return result;
}

unsigned char* filter_GPU(unsigned char* pixelData, int width, int height)
{
	size_t size = width * height;//размер фотки в кол-ве элементов
	unsigned char* pixelDataGPU, * resultGPU;//создаём входные и выходные матрицы на видеокарте

	cudaMalloc((void**)&pixelDataGPU, size);//выделяем память на видеокарте для входных данных
	cudaMalloc((void**)&resultGPU, size);//выделяем для выходных
	cudaMemcpy(pixelDataGPU, pixelData, size, cudaMemcpyHostToDevice);//копируем данные из процессора на видеокарту

	dim3 bl(8, 8);//создаём блок размер 8*8 нитей
	dim3 grid;
	//считаем кол-во блоков в гриде
	grid.x = height / bl.x;
	if (height % bl.x != 0)
		grid.x += 1;

	grid.y = width / bl.y;
	if (width % bl.y != 0)
		grid.y += 1;
	//засекаем время
	filter_kernel << <grid, bl >> > (pixelDataGPU, resultGPU, height, width);//запускаем ядро
	cudaDeviceSynchronize();
	unsigned char* result = new unsigned char[size];//создаем результирующую матрицу которую вернём в мейн
	cudaMemcpy(result, resultGPU, size, cudaMemcpyDeviceToHost);//копируем в неё данные из видеокарты на процессор
	cudaFree(pixelDataGPU);//очищаем память
	cudaFree(resultGPU);
	cudaDeviceReset();//выключаем работу с видеокартой
	return result;
}

char* uinttostr(unsigned int n) {
	int p;
	unsigned int n1;
	char* rv;

	p = 0;
	n1 = n;
	do {
		++p;
		n1 /= 10;
	} while (n1 > 0);
	rv = new char[p];
	rv[p] = 0;
	do {
		--p;
		rv[p] = '0' + n % 10;
		n /= 10;
	} while (n > 0);
	return rv;
}

char* ch[7] = {
	 (char*)"../images/1.pgm\0" ,
	 (char*)"../images/2.pgm\0" ,
	 (char*)"../images/3.pgm\0" ,
	 (char*)"../images/4.pgm\0" ,
	 (char*)"../images/5.pgm\0" ,
	 (char*)"../images/6.pgm\0" ,
	 (char*)"../images/7.pgm\0"
};

void MPIdca(int world_size, int world_rank, int* a)
{
	int index = 0;
	for (int i = 0; i < world_rank; i++)
	{
		index += a[i];
	}
	for (int i = index; i < index + a[world_rank]; i++)
	{
		unsigned char* img1 = NULL;
		unsigned char* rebuild = NULL;
		unsigned int w = 0;
		unsigned int h = 0;
		unsigned int chanels;
		string wayS;

		char* way = ch[i];
		__loadPPM(way, &img1, &w, &h, &chanels);
		
		unsigned char* img = rebuildSrcData(img1, w, h);
		w += 2;
		h += 2;

		unsigned char* imgCPU = NULL;
		unsigned char* imgGPU = NULL;

		imgCPU = filter_CPU(img, h, w);
	
		imgGPU = filter_GPU(img, h, w);

		/*imgCPU = returnData(imgCPU, w, h);
		imgGPU = returnData(imgGPU, w, h);

		w -= 2;
		h -= 2;*/

		cout << "Image " << world_rank << " size: " << h << "x" << w << endl;

#pragma region save

		wayS = "../images/CPU";
		wayS += uinttostr(i + 1);
		wayS += ".pgm";

		way = new char[wayS.size() + 1];
		copy(wayS.begin(), wayS.end(), way);
		way[wayS.size()] = '\0';

		__savePPM(way, imgCPU, w, h, 1);

		wayS = "../images/GPU";
		wayS += uinttostr(i + 1);
		wayS += ".pgm";

		way = new char[wayS.size() + 1];
		copy(wayS.begin(), wayS.end(), way);
		way[wayS.size()] = '\0';
		__savePPM(way, imgGPU, w, h, 1);
#pragma endregion
	}
}

int main(int argc, char** argv) {


	MPI_Init(NULL, NULL);
	int n = 7;

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	int* a = generate(world_size, n);
	for (int i = 0; i < world_size; i++)
		cout << a[i] << "  ";

	MPIdca(world_size, world_rank, a);
	MPI_Finalize();

}

inline unsigned int ind(unsigned int height, unsigned int width, unsigned int width_size) {
	return height * width_size + width;
}

void compose(unsigned char* matrix1, unsigned char* matrix2, int h, int w) {
	int t = 0;
	for (int i = 0; i < h * w; i++)
		if (matrix1[i] != matrix2[i]) {
			t++;
		}
	cout << "Count of miss: " << t << endl;
}

int* generate(int world_size, int n)
{
	int size = n / world_size;
	int* a = new int[world_size];
	for (int i = 0; i < world_size; i++)
	{
		a[i] = size;
	}
	if (n % world_size != 0)
		for (int i = 0; i < n % world_size; i++)
			a[i] += 1;
	return a;
}