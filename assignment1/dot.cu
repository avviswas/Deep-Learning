//These are the necessary libraries
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
//#include <cuda_runtime_api.h>
#include "dot.h"
// This is the main program
int main(int argc ,char* argv[]) {


FILE *data;
FILE *weights;
size_t size;
	

// Here we are declaring the rows and columns and CUDA device and number of threads 
unsigned int rows=atoi(argv[3]);
unsigned int cols=atoi(argv[4]);
int CUDA_DEVICE = atoi(argv[5]);
int THREADS = atoi(argv[6]);
printf("Put in values\n");
printf("Rows= %d\n,Cols = %d\n,CUDA_DEVICE= %d\n, THREADS =%d \n",rows,cols,CUDA_DEVICE,THREADS);
cudaError err = cudaSetDevice(CUDA_DEVICE);
if(err != cudaSuccess) { printf("Error in setting the CUDA device\n"); exit(EXIT_FAILURE); }



// Here we declare the variables for the host 
int BLOCKS;
float* host_results = (float*) malloc(rows * sizeof(float)); 
struct timeval starttime, endtime;
clock_t start, end;
float seconds = 0;
unsigned int jobs; 
unsigned long i;



// Here we declare the variables for the device
float  *dev_dataT;
float *dev_dataV;
float *results;
//size_t len = 0;
float arr[rows][cols];
float var ;
int vrow =1;
start = clock();



// Check to see if the files can be read in
data = fopen(argv[1], "r");
weights = fopen(argv[2],"r");
if (data == NULL) {
  printf("Error in reading in the data");
	return 0;
}
if (weights == NULL){
	printf("Error in reading in the weights");
}
size = (size_t)((size_t)rows*(size_t)cols);
size_t sizeV = 0;
sizeV = (size_t)((size_t)vrow*(size_t)cols);
//printf("The size of the data is %lu\n",size);
fflush(stdout);




// Here we allocate the memory for the data files that we are reading in above
float *dataT = (float*)malloc((size)*sizeof(float));
float *dataV = (float*)malloc((sizeV)*sizeof(float));
if(dataT == NULL) {
	printf("Error in allocating memory for the data file.\n");
}
if(dataV == NULL){
	printf("Error in allocating memory for the weights file. \n");
}  
gettimeofday(&starttime, NULL);
int j = 0;
// Here we are moving the data from the file to the allocated memory // on the CPU
	printf("Currentl moving the data from the files to the allocated memory on the CPU \n");
for (i =0; i< rows;i++){
	for(j=0; j<cols ; j++){
		fscanf(data,"%f",&var);
      arr[i][j]=var;
}
}
for (i =0;i<cols;i++){
  for(j= 0; j<rows; j++){
		dataT[rows*i+j]= arr[j][i];
}
}		
for (j=0;j<cols;j++){
	fscanf(weights,"%f",&dataV[j]);
}   
fclose(data);
fclose(weights);
printf("Reading data has been completed\n");
fflush(stdout);
gettimeofday(&endtime, NULL);
seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
printf("The time it took to read the data is %f\n", seconds);




// Here we are allocating memory on the GPU for the data
printf("Currently allocating memory on the GPU for the data\n");
gettimeofday(&starttime, NULL);
err = cudaMalloc((float**) &dev_dataT, (size_t) size * (size_t) sizeof(float));
if(err != cudaSuccess) { printf("Error in allocating memory on the GPU\n"); }
gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
printf("The time it took to allocate memory for the data is %f\n", seconds);
gettimeofday(&starttime, NULL);


// Here we do the same for the vector
err = cudaMalloc((float**) &dev_dataV, sizeV * sizeof(float));
if(err != cudaSuccess) { printf("Error in allocating memory on GPU\n"); }
gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
printf("The time it took to allocate memory for the weights is %f\n", seconds);
gettimeofday(&starttime, NULL);
	
// Here we are allocating memory on the GPU for the results
err = cudaMalloc((float**) &results, rows * sizeof(float) );
if(err != cudaSuccess) { printf("Error in allocating memory on the GPU for the results\n"); }
gettimeofday(&endtime, NULL); 
seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
printf("time for cudamalloc for result =%f\n", seconds);

// Here we are copying the data to the GPU
printf("Currently copying the data to the allocated memory in GPU\n");
gettimeofday(&starttime, NULL);
err = cudaMemcpy(dev_dataT, dataT, (size_t)size *sizeof(float), cudaMemcpyHostToDevice);
if(err != cudaSuccess) { printf("Error in copying data to the GPU\n"); }
gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
printf("The time it took to copy the data to the GPU is %f\n", seconds);

// Copying the weights to the allocated memory on the GPU
gettimeofday(&starttime, NULL);
err = cudaMemcpy(dev_dataV, dataV, sizeV*sizeof(float), cudaMemcpyHostToDevice);
if(err != cudaSuccess) { printf("Error in copying the weights to the GPU\n"); }
gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
printf("The time it took to copy the weights to the GPU is %f\n", seconds);
jobs = rows;
BLOCKS = (jobs + THREADS - 1)/THREADS;
gettimeofday(&starttime, NULL);

// This is where we finally call the kernel function
kernel<<<BLOCKS,THREADS>>>(rows,cols,dev_dataT,	dev_dataV, results);
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
printf("The time it took for the kernel is %f\n", seconds);
		
// We finally copy the results back to the CPU
cudaMemcpy(host_results,results,rows * sizeof(float),cudaMemcpyDeviceToHost);
printf("The dot product of the data and weights is \n");
printf("\n");

for(int k = 0; k < jobs; k++) {
	printf("%f ", host_results[k]);
	printf("\n");
}
printf("\n");
cudaFree( dev_dataT );
cudaFree( results );
end = clock();
seconds = (float)(end - start) / CLOCKS_PER_SEC;
printf("The time it took to run the whole program is %f\n", seconds);
return 0;
}
