#include <iostream>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <ratio>
#include <xmmintrin.h>
#include <immintrin.h>
#include <windows.h>
#pragma intrinsic(__rdtsc);
using namespace std;
using namespace chrono;
#define KB 1024
#define MB 1024 * KB 
#define OffsetL2 4 * MB / sizeof(lli)
#define Nmax 20
#define Cashe_size 4 * MB / sizeof(lli)

typedef unsigned long long int lli;

void init(lli* arr, int n) {

	ZeroMemory(arr, OffsetL2 * Nmax);

	if (n == 1) {
		for (size_t i = 0; i < Cashe_size - 1; i++)
		{
			arr[i] = i + 1;
		}return;
	}
		size_t blockSize = Cashe_size % n == 0 ? Cashe_size / n : Cashe_size / n + 1;		
		size_t currentOffsetL2 = 0;
		for (size_t i = 0; i < n-1; i++) {

			for (size_t j = 0; j < blockSize; j++)
				arr[currentOffsetL2 + j] = currentOffsetL2 + OffsetL2 + j;
			currentOffsetL2 += OffsetL2;//1->33; 2->34 3-
		}
		for (size_t i = 0; i < blockSize; i++)
			arr[currentOffsetL2 + i] = i + 1;
}

int main() {
	lli* arr;
	arr = (lli*)_aligned_malloc((OffsetL2 * Nmax) * sizeof(lli), 64);
	//arr = new lli[OffsetL2 * Nmax];
	for (int i = 2; i < Nmax; i++) {
		init(arr, i);
		size_t t = 0;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		for (size_t k = 0; k < 1000; k++)
		{
			do {
				t = arr[t];
			} while (t);
		}
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		cout << i << "---> " << time_span.count() << " seconds." << endl;
	}
	system("pause");
	return 0;
}