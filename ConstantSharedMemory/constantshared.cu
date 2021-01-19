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

#define TILE_WIDTH 16 // 256 threads
#define maskCols 5
#define maskRows 5
#define BLOCK_WIDTH (TILE_WIDTH + maskCols -1)

//mask in constant memory
__constant__ float deviceMaskData[maskRows * maskCols];
__global__ void ConstantSharedMemoryConvolution(float * InputImageData, const float *__restrict__ kernel,
		float* outputImageData, int channels, int width, int height){

	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];	//block of image in shared memory


	// allocation in shared memory of image blocks
	int maskRadius = maskRows/2;
 	for (int k = 0; k <channels; k++) {
 		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x; // flatten the 2D coordinates of the generic thread
 		int destY = dest/BLOCK_WIDTH;     //col of shared memory
 		int destX = dest%BLOCK_WIDTH;		//row of shared memory
 		int srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;  //row index to fetch data from input image
 		int srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;	//col index to fetch data from input image
 		if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
 			N_ds[destY][destX] = InputImageData[(srcY *width +srcX) * channels + k]; // copy element of image in shared memory
 		else
 			N_ds[destY][destX] = 0;


 		dest = threadIdx.y * TILE_WIDTH+ threadIdx.x + TILE_WIDTH * TILE_WIDTH;
 		destY = dest/BLOCK_WIDTH;
		destX = dest%BLOCK_WIDTH;
		srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;
		srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;
		if(destY < BLOCK_WIDTH){
			if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
				N_ds[destY][destX] = InputImageData[(srcY *width +srcX) * channels + k];
			else
				N_ds[destY][destX] = 0;
		}

 		__syncthreads();


 		//compute kernel convolution
 		float Pvalue = 0;
 		int y, x;
 		for (y= 0; y < maskCols; y++)
 			for(x = 0; x<maskRows; x++)
 				Pvalue += N_ds[threadIdx.y + y][threadIdx.x + x] *deviceMaskData[y * maskCols + x];

 		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
 		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
 		if(y < height && x < width)
 			outputImageData[(y * width + x) * channels + k] = Pvalue;
 		__syncthreads();


 	}

}
 	
void imageConvolutionConstantSharedMemory(const char* inputfilepath, const char* outputfilepath ){

	int imgChannels;
	int imgHeight;
	int imgWidth;
	Image* inputImage;
	Image* outputImage;
	float* hostInputImageData;
	float* hostOutputImageData;
	float* deviceInputImageData;
	float* deviceOutputImageData;
	float hostMaskData[maskRows * maskCols]={
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06
	};


	inputImage = import_PPM(inputfilepath);

	imgWidth = img_getWidth(inputImage);
	imgHeight = img_getHeight(inputImage);
	imgChannels = img_getChannels(inputImage);

	outputImage = Image_new(imgWidth, imgHeight, imgChannels);

	hostInputImageData = img_getData(inputImage);
	hostOutputImageData = img_getData(outputImage);

	cudaDeviceReset();
	cudaMalloc((void **) &deviceInputImageData, imgWidth * imgHeight *
			imgChannels * sizeof(float));
	cudaMalloc((void **) &deviceOutputImageData, imgWidth * imgHeight *
			imgChannels * sizeof(float));
	cudaMemcpy(deviceInputImageData, hostInputImageData,
			imgWidth * imgHeight * imgChannels * sizeof(float),
			cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(deviceMaskData, hostMaskData, maskRows * maskCols * sizeof(float));

	float numberBlockXTiling = (float) imgWidth / TILE_WIDTH;
	float numberBlockYTiling = (float) imgHeight / TILE_WIDTH;

	int numberBlockX = ceil(numberBlockXTiling);
	int numberBlockY = ceil(numberBlockYTiling);

	dim3 dimGrid(numberBlockX, numberBlockY);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);


	cout << "CONVOLUTION CONSTANT SHARED MEMORY" << endl;
    cout << "Image dimensions : " << imgWidth << "x" << imgHeight << " , Channels : " << imgChannels << endl;
	cout << "Time: ";
	high_resolution_clock::time_point start= high_resolution_clock::now();

	ConstantSharedMemoryConvolution<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
	imgChannels, imgWidth, imgHeight);

	high_resolution_clock::time_point end= high_resolution_clock::now();
	chrono::duration<double>  duration = end - start;
	cout << duration.count()*1000 << " millisec" <<endl;
	cout << "----------------------------------" << endl;

	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imgWidth * imgHeight *
			imgChannels * sizeof(float), cudaMemcpyDeviceToHost);

	write_image(outputfilepath, outputImage);

	cudaMemset(deviceInputImageData,0,imgWidth * imgHeight *
				imgChannels * sizeof(float));
	cudaMemset(deviceOutputImageData,0,imgWidth * imgHeight *
					imgChannels * sizeof(float));
	cudaMemset(deviceMaskData,0,maskRows * maskCols
				* sizeof(float));
				
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	Image_delete(outputImage);
	Image_delete(inputImage);
	
}

int main() {

	imageConvolutionConstantSharedMemory("/home/aventuri/progetto/constantshared/photoSD.ppm","/home/aventuri/progetto/constantshared/resultSDSC.ppm");
	imageConvolutionConstantSharedMemory("/home/aventuri/progetto/constantshared/photoHD1.ppm","/home/aventuri/progetto/constantshared/resultHD1SC.ppm");
	imageConvolutionConstantSharedMemory("/home/aventuri/progetto/constantshared/photoHD2.ppm","/home/aventuri/progetto/constantshared/resultHD2SC.ppm");
	imageConvolutionConstantSharedMemory("/home/aventuri/progetto/constantshared/photo4K.ppm","/home/aventuri/progetto/constantshared/result4KSC.ppm");

}


 	
