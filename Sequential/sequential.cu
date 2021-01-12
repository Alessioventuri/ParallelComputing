/*
 * main.cpp
 *
 *  Created on: Jan 1, 2021
 *      Author: alessioventuri
 */



#include "Image.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <time.h>
#include <iostream>
#include "functionImage.h"

using namespace std;
using namespace std::chrono;

void sequentialConvolution(Image* inputImage,float* outputImage,int channels,int imgWidth,
		int imgHeight,int kernelWidth, int kernelHeight,const float* kernel){

	int i,j,k;
	float Pvalue ;
	int kernelCenterX, kernelCenterY;

	kernelCenterX = kernelWidth/2;
	kernelCenterY = kernelHeight/2;

	float* dataImageInput = img_getData(inputImage);

	for(k = 0; k < channels; ++k){

		for(i = 0; i < imgHeight ; ++i){

			for(j = 0; j < imgWidth ; ++j){
				Pvalue = 0;

				for ( int x = 0; x < kernelHeight; ++x){

					int rowIndex = kernelHeight - 1 - x; //row index from end kernel to start

					for (int y = 0; y < kernelWidth; ++y){

						int colIndex = kernelWidth - 1 - y; // col index from end kernel to start

						int kernelBoundRow = i + x - kernelCenterY;
						int kernelBoundCol = j + y - kernelCenterX;

						if(kernelBoundRow >= 0 && kernelBoundRow < imgHeight && kernelBoundCol >= 0 && kernelBoundCol < imgWidth ){
							Pvalue += dataImageInput[(imgWidth * kernelBoundRow+kernelBoundCol)* channels + k]* kernel[kernelWidth*rowIndex + colIndex];
						}
					}
				}
				outputImage[(imgWidth*i + j)*channels + k] = Pvalue;
			}
		}
	}

}



void ImageSequentialConvolution(const char* inputfilepath, const char* outputfilepath){

	Image* inputImage;
	int imgChannels;
	int imgWidth;
	int imgHeight;
	Image* outputImage;
	float* outputImageData;

	const int kernelWidth = 5;
	const int kernelHeight = 5;
	const float kernel [kernelWidth * kernelHeight] =  {
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06,
			0.06, 0.06, 0.06, 0.06, 0.06
    };

	inputImage = import_PPM(inputfilepath);
	imgChannels = img_getChannels(inputImage);
	imgWidth = img_getWidth(inputImage);
	imgHeight = img_getHeight(inputImage);
	outputImageData = img_getData(inputImage);
	outputImage = inputImage;

	cout << "SEQUENTIAL CONVOLUTION " << endl;
    cout << "Image dimensions : " << imgWidth << "x" << imgHeight << " , Channels : " << imgChannels << endl;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    sequentialConvolution(inputImage, outputImageData, imgChannels, imgWidth, imgHeight, kernelWidth, kernelHeight, kernel);
	high_resolution_clock::time_point endSeq = high_resolution_clock::now();
	auto durationSeq = (double) duration_cast<milliseconds>(endSeq - start).count() / 1000;
	cout << "Time: " << durationSeq << endl;
	img_setData(outputImage, outputImageData);
	write_image(outputfilepath, outputImage);
	cout << "-------------------------" << endl;

}

int main(){
	ImageSequentialConvolution("/home/aventuri/progetto/sequential/photoSD.ppm","/home/aventuri/progetto/sequential/resultSDS.ppm");
	ImageSequentialConvolution("/home/aventuri/progetto/sequential/photoHD1.ppm","/home/aventuri/progetto/sequential/resultHD1S.ppm");
	ImageSequentialConvolution("/home/aventuri/progetto/sequential/photoHD2.ppm","/home/aventuri/progetto/sequential/resultHD2S.ppm");
	ImageSequentialConvolution("/home/aventuri/progetto/sequential/photo4K.ppm","/home/aventuri/progetto/sequential/result4KS.ppm");
}










