#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dot.h"

void kernel(unsigned int rows, unsigned int cols , float* ddata,float* vdata ,float *results, unsigned int jobs){
	
	int i, j, stop;
        float dp;
        int tid = omp_get_thread_num();
        if ((tid +1)*jobs > rows) stop = rows;
        else stop = (tid +1)*jobs;
        printf("thread id= %d, start = %d, stop =%d \n",tid,tid*jobs, stop);

	for (j = tid*jobs; j<stop; j++)
        {
	    dp = 0;
            for(i =0; i<cols ;i++ )
            {
//		    printf("jobs:%d, tid:%d, i:%d, j:%d\n", jobs, tid, i, j);
                    dp+= ddata[(size_t) ((size_t) i) * ((size_t) rows)+j]*vdata[i];
//		   printf("D Data:%f\n", ddata[(size_t) ((size_t) i) * ((size_t) rows)+j]);
//		   printf("V Data:%f\n", vdata[i]);		
            }
            results[j] = dp;
        }
}

