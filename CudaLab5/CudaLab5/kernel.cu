#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <stdio.h>
#include <windows.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include "Shlwapi.h"

#include "helper_image.h"

unsigned char* rebuildSrcData(unsigned char* srcData, int width, int height);
unsigned char* returnData(unsigned char* Data, int width, int height);
unsigned char filter(unsigned char* pixelData, int i, int j, int w);

__forceinline__ __device__ unsigned char filterCUDA(unsigned char* pixelData, int i, int j, int w)
{
	unsigned char result;

	result = pixelData[(i - 1) * w + j] + pixelData[i * w + (j - 1)] - pixelData[i * w + (j + 1)] - pixelData[(i + 1) * w + j] + 128;

	if (result < 0) result = 0;
	if (result > 255) result = 255;

	return result;
}

inline unsigned int ind(unsigned int height, unsigned int width, unsigned int width_size);

__global__ void filter_kernel(unsigned char* inputBitmap, unsigned char* outputBitmap, int height, int width) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex > 0) && (yIndex > 0))
	{
		//тупо берём и накладываем фильтр
		outputBitmap[xIndex * width + yIndex] = filterCUDA(inputBitmap, xIndex, yIndex, width);
	}
}

//накладываем фильтр с помощью процессора
unsigned char* filter_CPU(unsigned char* pixelData, int w, int h) {
	LARGE_INTEGER start, finish, freq;//опять же время
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	unsigned char* result = new unsigned char[w * h];//создаём результирующую матрицу
	if (result == NULL)
		return NULL;
	//тупо поэлементно накладываем фильтр
	for (int i = 1; i < h - 1; i++) {
		for (int j = 1; j < w - 1; j++) {
			result[(i)*w + (j)] = filter(pixelData, i, j, w);
		}
	}

	QueryPerformanceCounter(&finish);
	double time = (finish.QuadPart - start.QuadPart) / (double)freq.QuadPart;
	printf("\ntime CPU = %lf\n", time);
	return result;
}
//накладываем фильтр с помощью видеокарты
unsigned char* filter_GPU(unsigned char* pixelData, int width, int height)
{
	size_t size = width * height;//размер фотки в кол-ве элементов
	LARGE_INTEGER start, finish, freq;//для времени
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
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	filter_kernel << <grid, bl >> > (pixelDataGPU, resultGPU, height, width);//запускаем ядро
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&finish);
	double time = (finish.QuadPart - start.QuadPart) / (double)freq.QuadPart;
	printf("\ntime CUDA = %lf\n", time);
	unsigned char* result = new unsigned char[size];//создаем результирующую матрицу которую вернём в мейн
	cudaMemcpy(result, resultGPU, size, cudaMemcpyDeviceToHost);//копируем в неё данные из видеокарты на процессор
	cudaFree(pixelDataGPU);//очищаем память
	cudaFree(resultGPU);
	cudaDeviceReset();//выключаем работу с видеокартой
	return result;
}
//сравниваем две картинки(одна после обработки на процессоре, вторая на видеокарте)
bool isEquals(unsigned char* a, unsigned char* b, int width, int height) {
	//тупо сравниваем по элементно
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			if (a[i + j * width] != b[i + j * width]) {
				return false;
			}
	return true;
}


int main() {
	unsigned int width = 0, height = 0, channels;
	const char srcImage[] = "D:\\6 сем\\АВП\\CudaLab5\\CudaLab5\\7.pgm";//пути для хранения фотографий
	const char imageCPU[] = "D:\\6 сем\\АВП\\CudaLab5\\CudaLab5\\imageCPU2.pgm";
	const char imageGPU[] = "D:\\6 сем\\АВП\\CudaLab5\\CudaLab5\\imageGPU2.pgm";
	unsigned char* srcData = NULL, * GPUData = NULL, * CPUData = NULL; //объявляем переменные для хранения фото
															 // srcData-исходная   GPUData-после обработки на видеокарте
															 //CPUData- после обработки на процессоре
	__loadPPM(srcImage, &srcData, &width, &height, &channels);

	unsigned char* dataRebuild = rebuildSrcData(srcData, width, height);//создаём границы фотографии из 0

	width += 2;//из-за добавления границ увеличилась ширина и высота фото
	height += 2;

	CPUData = filter_CPU(dataRebuild, width, height);//обрабатываем на процессоре
	GPUData = filter_GPU(dataRebuild, width, height);//на видеокарте

	CPUData = returnData(CPUData, width, height);//убираем границы
	GPUData = returnData(GPUData, width, height);

	width -= 2;//из-за удалений границ уменьшились ширина и высота фото
	height -= 2;
	//сравниваем две фотографии, если isEquals = true выводим equals, в противном случае - not equals
	isEquals(CPUData, GPUData, width, height) ? printf("\nequals\n") : printf("\nnot equals\n");//сравниваем

	__savePPM(imageCPU, CPUData, width, height, channels);//сохранеям фотки
	__savePPM(imageGPU, GPUData, width, height, channels);

	free(srcData);//очищаем память
	free(GPUData);
	free(CPUData);

	system("pause");

	return 0;
}

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

unsigned char filter(unsigned char* pixelData, int i, int j, int w)
{
	unsigned char result;

	result = pixelData[(i - 1) * w + j] + pixelData[i * w + (j - 1)] - pixelData[i * w + (j + 1)] - pixelData[(i + 1) * w + j] + 128;

	if (result < 0) result = 0;
	if (result > 255) result = 255;

	return result;
}

inline unsigned int ind(unsigned int height, unsigned int width, unsigned int width_size) {
	return height * width_size + width;
}//подсчёт индекса