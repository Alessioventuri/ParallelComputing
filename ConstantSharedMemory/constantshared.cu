#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>

#include "device_launch_parameters.h"
#include "functionImage.h"


using namespace std::chrono;
using namespace std;


#define TILE_WIDTH 16  // 256 threads
#define maskCols 5
#define maskRows 5
#define BLOCK_WIDTH (TILE_WIDTH + maskCols -1)
#define clamp(x) (min(max((x), 0.0), 1.0))

// mask in constant memory
__constant__ float deviceMaskData[maskRows * maskCols];

__global__ void ConstantSharedMemoryConvolution(float * InputImageData, const float *__restrict__ kernel,
		float* outputImageData, int channels, int width, int height){

	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];  //block of image in shared memory
	
	// allocation in shared memory of image blocks
	
	int maskRadius = maskRows/2;
 	for (int k = 0; k <channels; k++) {
 		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x; // flatten the 2D coordinates of the generic thread
 		int destY = dest/BLOCK_WIDTH;   //row of shared memory (makes the inverse operation , in that it calculates the 2D coordinates )
 		int destX = dest%BLOCK_WIDTH;	//col of shared memory (of the generica thread with respect to the shared memory area )	
 		int srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius; // index to fetch data from input image
 		int srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius; // index to fetch data from input image
 		int src = (srcY *width +srcX) * channels + k;   // index of input image
 		
 		// When a thread is to load any input element, test if it is in the valid index range
 		
 		if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
 			N_ds[destY][destX] = InputImageData[src];  // copy element of image in shared memory
 		else
 			N_ds[destY][destX] = 0;

 		dest = threadIdx.y * TILE_WIDTH+ threadIdx.x + (TILE_WIDTH * TILE_WIDTH);
 		destY = dest/BLOCK_WIDTH;
		destX = dest%BLOCK_WIDTH;
		srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;
		srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;
		src = (srcY *width +srcX) * channels + k;
		if(destY < BLOCK_WIDTH){
			if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
				N_ds[destY][destX] = InputImageData[src];
			else
				N_ds[destY][destX] = 0;
		}

		//Barrier synchronization
 		__syncthreads();


 		//compute kernel convolution
 		float Pvalue = 0;
 		int y, x;
 		for (y= 0; y < maskCols; y++)
 			for(x = 0; x<maskRows; x++)
 				Pvalue += N_ds[threadIdx.y + y][threadIdx.x + x] * deviceMaskData[y * maskCols + x];

 		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
 		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
 		if(y < height && x < width)
 			outputImageData[(y * width + x) * channels + k] = clamp(Pvalue);
 		__syncthreads();


	}
}
 	
void imageConvolutionConstantSharedMemory(const char* inputfilepath, const char* outputfilepath ){

	int imgChannels;
	int imgHeight;
	int imgWidth;
	Image* imgInput;
	Image* imgOutput;
	float* hostInputImageData;
	float* hostOutputImageData;
	float* deviceInputImageData;
	float* deviceOutputImageData;
	float* deviceMaskData;
	float hostMaskData[maskRows * maskCols]={
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06

	};


	imgInput = import_PPM(inputfilepath);
	
    imgWidth = img_getWidth(imgInput);
    imgHeight = img_getHeight(imgInput);
    imgChannels = img_getChannels(imgInput);

	imgOutput = Image_new(imgWidth, imgHeight, imgChannels);

    hostInputImageData = img_getData(imgInput);
    hostOutputImageData = img_getData(imgOutput);

	cudaDeviceReset();
	cudaMalloc((void **) &deviceInputImageData, imgWidth * imgHeight *
			imgChannels * sizeof(float));
			
	cudaMalloc((void **) &deviceOutputImageData, imgWidth * imgHeight *
			imgChannels * sizeof(float));
			
	cudaMalloc((void **) &deviceMaskData, maskRows * maskCols
			* sizeof(float));
			
	cudaMemcpy(deviceInputImageData, hostInputImageData,
			imgWidth * imgHeight * imgChannels * sizeof(float),
			cudaMemcpyHostToDevice);
	
	// Copies "count" bytes from the memory area pointed to by hostMask to the memory area pointed to by "offset" bytes from 
    // the start of symbol "deviceMask"
	cudaMemcpyToSymbol(deviceMaskData, hostMaskData, maskRows * maskCols * sizeof(float));


	dim3 dimGrid(ceil((float) imgWidth/TILE_WIDTH),
			ceil((float) imgHeight/TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);


	cout << "CONVOLUTION CONSTANT SHARED MEMORY" << endl;
    cout << "Image dimensions : " << imgWidth << "x" << imgHeight << " , Channels : " << imgChannels << endl;
	cout << "Time: ";
	high_resolution_clock::time_point start= high_resolution_clock::now();

	ConstantSharedMemoryConvolution<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
	imgChannels, imgWidth, imgHeight);

	high_resolution_clock::time_point end= high_resolution_clock::now();
	chrono::duration<double>  duration = end - start;
	cout << duration.count()*1000 << endl;
	cout << "----------------------------------" << endl;

	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imgWidth * imgHeight *
			imgChannels * sizeof(float), cudaMemcpyDeviceToHost);

	write_image(outputfilepath, imgOutput);

	cudaMemset(deviceInputImageData,0,imgWidth * imgHeight *
				imgChannels * sizeof(float));
	cudaMemset(deviceOutputImageData,0,imgWidth * imgHeight *
					imgChannels * sizeof(float));
	cudaMemset(deviceMaskData,0,maskRows * maskCols
				* sizeof(float));
				
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	Image_delete(imgOutput);
	Image_delete(imgInput);
	
}

int main() {

	imageConvolutionConstantSharedMemory("/home/aventuri/progetto/constantshared/photoSD.ppm","/home/aventuri/progetto/constantshared/resultSDSC.ppm");
	imageConvolutionConstantSharedMemory("/home/aventuri/progetto/constantshared/photoHD1.ppm","/home/aventuri/progetto/constantshared/resultHD1SC.ppm");
	imageConvolutionConstantSharedMemory("/home/aventuri/progetto/constantshared/photoHD2.ppm","/home/aventuri/progetto/constantshared/resultHD2SC.ppm");
	imageConvolutionConstantSharedMemory("/home/aventuri/progetto/constantshared/photo4K.ppm","/home/aventuri/progetto/constantshared/result4KSC.ppm");

}


 	
