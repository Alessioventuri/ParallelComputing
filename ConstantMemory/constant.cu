	#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>

#include "functionImage.h"
#include "device_launch_parameters.h"


using namespace std::chrono;
using namespace std;

#define maskCols 5
#define maskRows 5

// mask in constant memory
__constant__ float deviceMaskData[maskRows * maskCols];

__global__ void constantMemoryConvolution(float * InputImageData, const float *__restrict__ kernel,
		float* outputImageData, int channels, int width, int height){

	float Pvalue = 0;
	
	int col = threadIdx.x + blockIdx.x * blockDim.x; //number of threads along x axis
	int row = threadIdx.y + blockIdx.y * blockDim.y; //number of threads along y axis
	int maskRowsRadius = maskRows/2;
	int maskColsRadius = maskCols/2;

	for (int k = 0; k < channels; k++){    
		if(row < height && col < width ){
			Pvalue = 0;
			int startRow = row - maskRowsRadius;    
			int startCol = col - maskColsRadius;	
			
			for(int i = 0; i < maskRows; i++){	

				for(int j = 0; j < maskCols; j++){	

					int currentRow = startRow + i;	
					int currentCol = startCol + j;	

					if(currentRow > -1 && currentRow < height && currentCol > -1 && currentCol < width){ // Check the unused threads

							Pvalue += InputImageData[(currentRow * width + currentCol )*channels + k] *
										deviceMaskData[i * maskRows + j];
					}
					else Pvalue = 0;
				}

			}
			outputImageData[(row* width + col) * channels + k] = Pvalue;
		}
			
	}
	
}

void imageConvolutionConstantMemory(const char* inputfilepath, const char* outputfilepath ){

	int imgChannels;
    int imgWidth;
    int imgHeight;
    Image* imgInput;
    Image* imgOutput;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* deviceInputImageData;
    float* deviceOutputImageData;
    float maskData[maskRows * maskCols] = {
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

    // copies "count" bytes from the memory area pointed to by hostInputImageData to the memory area pointer to by deviceInputImageData
  	cudaMemcpy(deviceInputImageData, hostInputImageData,
			imgWidth * imgHeight * imgChannels * sizeof(float),
			cudaMemcpyHostToDevice);
			
    // Copies "count" bytes from the memory area pointed to by hostMask to the memory area pointed to by "offset" bytes from 
    // the start of symbol "deviceMask"
    cudaMemcpyToSymbol(deviceMaskData, maskData, maskRows * maskCols * sizeof(float));

    dim3 dimGrid(ceil((float)imgWidth / 16),
        ceil((float)imgHeight / 16));  			// N	 thread blocks
        
	dim3 dimBlock(16, 16, 1); 					//16x16 thread per block
    
    cout << "CONVOLUTION CONSTANT MEMORY" << endl;
    cout << "Image dimensions : " << imgWidth << "x" << imgHeight << " , Channels : " << imgChannels << endl;
    high_resolution_clock::time_point start = high_resolution_clock::now();

								// N x 256 threads
    constantMemoryConvolution <<<dimGrid, dimBlock >>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
        imgChannels, imgWidth, imgHeight);

    high_resolution_clock::time_point end = high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time: " << duration.count() * 1000 << " millisec"<< endl;
    cout << "----------------------------------" << endl;

    // copies "count" bytes from the memory area pointed to by deviceOutputImageData to the memory area pointer to by hostOutputImageData
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,static_cast<unsigned long long>(imgWidth) * imgHeight *
        imgChannels * sizeof(float), cudaMemcpyDeviceToHost);

    write_image(outputfilepath, imgOutput);

	//Fills the first "count" bytes of the memory area pointed to by "deviceInputImageData" with the constant byte value "0"
	size_t count = static_cast<unsigned long long>(imgWidth) * imgHeight * imgChannels * sizeof(float);
	
    cudaMemset(deviceInputImageData, 0, count);
    cudaMemset(deviceOutputImageData, 0, count);
    cudaMemset(deviceMaskData, 0, maskRows * maskCols * sizeof(float));
    
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);
    
    Image_delete(imgInput);
    Image_delete(imgOutput);
    
}

int main() {

	imageConvolutionConstantMemory("/home/aventuri/progetto/constantmemory/photoSD.ppm","/home/aventuri/progetto/constantmemory/resultSDCM.ppm");
	imageConvolutionConstantMemory("/home/aventuri/progetto/constantmemory/photoHD1.ppm","/home/aventuri/progetto/constantmemory/resultHD1CM.ppm");
	imageConvolutionConstantMemory("/home/aventuri/progetto/constantmemory/photoHD2.ppm","/home/aventuri/progetto/constantmemory/resultHD2CM.ppm");
	imageConvolutionConstantMemory("/home/aventuri/progetto/constantmemory/photo4K.ppm","/home/aventuri/progetto/constantmemory/result4KCM.ppm");

}

