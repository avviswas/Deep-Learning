#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "dot.h"

__global__ void kernel(unsigned int rows, unsigned int cols , float* ddata,float* vdata ,float *results){
	
	int i;
        float dp =0;
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(i =0; i<cols ;i++ )
	{
		dp+= ddata[i*rows+tid]*vdata[i];		
	}
	
	results[tid] = dp;
	
}
