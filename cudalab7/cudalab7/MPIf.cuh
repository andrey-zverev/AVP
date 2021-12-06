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