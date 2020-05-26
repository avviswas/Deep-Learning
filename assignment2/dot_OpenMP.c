#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
//#include <cuda_runtime_api.h>
#include "dot.h"

int main(int argc ,char* argv[]) {
	
	FILE *data_file;
	FILE *vector_file;
	size_t size;
	
	/* Initialize rows, cols, CUDA devices and threads from the user */
	unsigned int rows=atoi(argv[3]);
	unsigned int cols=atoi(argv[4]);
	int nprocs = atoi(argv[5]);
	
	printf("\n*********** Input Values ***********\n\n");
	printf("Rows = %d\nCols = %d\nProcesses = %d \n",rows,cols,nprocs);

	/*Host variable declaration */

	float* host_results = (float*) malloc(rows * sizeof(float)); 
	struct timeval starttime, endtime;
	clock_t start, end;
	float seconds = 0;
	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
	
	float arr[rows][cols];
	float var ;
	int vrow =1;

	start = clock();

	/* Validation to check if the data file is readable */
	
	data_file = fopen(argv[1], "r");
	vector_file = fopen(argv[2],"r");
	
	if (data_file == NULL) {
    		printf("Cannot Open the File");
		return 0;
	}
	if (vector_file == NULL){
		printf("cannot open the file");
	}
	size = (size_t)((size_t)rows * (size_t)cols);
	size_t sizeV = 0;
	sizeV = (size_t)((size_t)vrow*(size_t)cols);

	/*printf("Size of the data = %lu\n",size);*/

	fflush(stdout);
	
	float *dataT = (float*)malloc((size)*sizeof(float));
	float *dataV = (float*)malloc((sizeV) * sizeof(float));

	if(dataT == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
	}
	
	if(dataV == NULL){
		printf("ERROR: Memory for vector not allocated. \n");
	}
        gettimeofday(&starttime, NULL);
	int j = 0;

    /* Transfer the Data from the file to CPU Memory */
	
	printf("\n*********** Transfer Data from File to CPU Memory ***********\n\n");
	
        for (i =0; i< rows;i++){
		for(j=0; j<cols ; j++){
			fscanf(data_file,"%f",&var);
                        arr[i][j]=var;
			//printf("%f\n",var);
		}
	}
//	printf("1 Done");
	for (i =0;i<cols;i++){
		for(j= 0; j<rows; j++){
			dataT[rows*i+j]= arr[j][i];
//			printf("Data: %f\n", arr[j][i]);
	}
	}		

		for (j=0;j<cols;j++){
			fscanf(vector_file,"%f",&dataV[j]);
//			printf("Vector: %f\n", dataV[j]);
		}
//   	printf("Read Data");
	fclose(data_file);
	fclose(vector_file);
        fflush(stdout);

        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        printf("time to read data = %f\n", seconds);

	jobs = (unsigned int) ((rows+nprocs-1)/nprocs);

        gettimeofday(&starttime, NULL);

	/* Calling the kernel function */
	
	printf("jobs=%d\n", jobs);
	
	kernel(rows,cols,dataT,	dataV, host_results, jobs);
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for kernel=%f\n", seconds);
			
	printf("\n*********** Output ***********\n\n");
	printf("\n");
	
	int k;
	
	for(k = 0; k < rows; k++) {
		printf("%f ", host_results[k]);
		printf("\n");
	}
	printf("\n");

	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Total time = %f\n", seconds);

	return 0;

}
