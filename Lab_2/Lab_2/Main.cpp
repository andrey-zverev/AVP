#include <iostream>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <ratio>
#include <xmmintrin.h>
#include <immintrin.h>

#define size 2048

void SSEMatrixMult(float** MTRX1, float** MTRX2, float** RESMTRX3);
void SSEMatrixMultBlock(float** MTRX1, float** MTRX2, float** RESMTRX3);

using namespace std;
using namespace chrono;

int main()
{
#pragma region MEMORY

	float** matrix_in_0, ** matrix_in_1, ** matrix_out_0, ** matrix_out_1;
	srand(static_cast <unsigned> (time(0)));
			matrix_in_0 = (float**)_aligned_malloc(size * sizeof(float*), 16);
			matrix_in_1 = (float**)_aligned_malloc(size * sizeof(float*), 16);
			matrix_out_0 = (float**)_aligned_malloc(size * sizeof(float*), 16);
			matrix_out_1 = (float**)_aligned_malloc(size * sizeof(float*), 16);

			for (int i = 0; i < size; i++) {
				matrix_in_0[i] = (float*)_aligned_malloc(size * sizeof(float), 16);
				matrix_out_0[i] = (float*)_aligned_malloc(size * sizeof(float), 16);
				matrix_out_1[i] = (float*)_aligned_malloc(size* sizeof(float), 16);

				for (int j = 0; j < size; j++) {
					matrix_in_0[i][j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 100));
				}

				for (int j = 0; j < size; j++) {
					matrix_out_0[i][j] = 0.0f;
					matrix_out_1[i][j] = 0.0f;
				}
			}

			for (int i = 0; i < size; i++) {
				matrix_in_1[i] = (float*)_aligned_malloc(size * sizeof(float), 16);

				for (int j = 0; j < size; j++) {
					matrix_in_1[i][j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 100));
				}
			}
#pragma endregion
	
	//sse
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	SSEMatrixMult(matrix_in_0, matrix_in_1, matrix_out_0);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "SSE " << time_span.count() << " seconds." << endl;

	t1 = high_resolution_clock::now();
	SSEMatrixMultBlock(matrix_in_0, matrix_in_1, matrix_out_1);
	t2 = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(t2 - t1);
	cout << "SSE_Block " << time_span.count() << " seconds." << endl;
	system("pause");

	for (int i = 0; i < size; i++) { 
		for (int j = 0; j < size; j++) {
			if (matrix_out_0[i][j] != matrix_out_1[i][j])printf("1");
		}
	}
	return 0;
}


void SSEMatrixMult(float** MTRX1, float** MTRX2, float** RESMTRX3) {

	__m256 M1, M2, RES3;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j <size; j += 8) {
			RES3 = _mm256_load_ps(&RESMTRX3[i][j]);
			for (int iin = 0; iin < size; iin++) {
				M1 = _mm256_set1_ps(MTRX1[i][iin]);
				M2 = _mm256_load_ps(&MTRX2[iin][j]);
				RES3 = _mm256_add_ps(RES3, _mm256_mul_ps(M1, M2));
			}
			_mm256_store_ps(&RESMTRX3[i][j], RES3);
		}
	}
}

void SSEMatrixMultBlock(float** MTRX1, float** MTRX2, float** RESMTRX3) {

	__m256 M1, M2, RES3;
	int block_size = 64;
	for (int ii = 0; ii < size; ii += block_size) {
		for (int jj = 0; jj < size; jj += block_size) {
			for (int in = 0; in < size; in += block_size)
			{
				int i_start = ii;    // индекс i для блока принимает значения [ii, ii + block_size)
				int i_end = ii + block_size;
				int j_start = jj;    // индекс j для блока принимает значения [jj, jj + block_size)
				int j_end = jj + block_size;
				int iin_start = in;
				int iin_end = in + block_size;

				// обходим блок
				for (int rowin = i_start; rowin < i_end; rowin++) {
					for (int colin = j_start; colin < j_end; colin += 8) {
						RES3 = _mm256_load_ps(&RESMTRX3[rowin][colin]);
						for (int iin = iin_start; iin < iin_end; iin++) {
							M1 = _mm256_set1_ps(MTRX1[rowin][iin]);
							M2 = _mm256_load_ps(&MTRX2[iin][colin]);
							RES3 = _mm256_add_ps(RES3, _mm256_mul_ps(M1, M2));
						}
						_mm256_store_ps(&RESMTRX3[rowin][colin], RES3);
					}
				}
			}
		}
	}
}