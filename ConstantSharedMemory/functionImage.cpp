#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "functionImage.h"

using namespace std;

char* File_read(FILE* file, size_t size, size_t count) {
    size_t res;
    char* buffer;
    size_t bufferLen;

    if (file == NULL) {
        return NULL;
    }

    bufferLen = size * count + 1;
    buffer = (char*)malloc(sizeof(char) * bufferLen);

    res = fread(buffer, size, count, file);
    // make valid C string
    buffer[size * res] = '\0';

    return buffer;
}
void retrieve_comment(char* buffer, int bufferLen, FILE* file){
	do{
		char c;
		buffer[0] = '\0';

		while( isspace ( c = getc(file))){};
		ungetc(c, file);
		if ('#' == c)
			fgets(buffer,bufferLen,file);
    	if (buffer[0] == '#')
    			printf("comment : %s\n", buffer);
	}while(strcmp(&buffer[0],"\0"));
	return;
}

Image* import_PPM(const char* filename) {

    FILE* file;
    char magicnum[3];
    unsigned width, height, maxval = 0;
    unsigned char* charData, * charIter;
    int channels = 0;
    Image* img;
    float* imgData, * floatIter;
    float scale;
    int x, y, z;
    char comment[4096];


    img = NULL;

    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Could not open %s\n", filename);
    }

    //Find magic number
    fscanf(file, "%2s", magicnum);

    if (strcmp(magicnum, "P5") == 0 || strcmp(magicnum, "P5\n") == 0) {
        channels = 1;
    }
    else if (strcmp(magicnum, "P6") == 0 || strcmp(magicnum, "P6\n") == 0) {
        channels = 3;
    }else
    	printf("Could not find magic number\n");

    //retrieve Comments
	retrieve_comment(comment, 4096,file);

    //find width and height
    fscanf(file, "%u %u", &width,&height);
	retrieve_comment(comment, 4096,file);
    fscanf(file, "%u", &maxval);
	retrieve_comment(comment, 4096,file);

    charData = (unsigned char*)File_read(file, width * channels * sizeof(unsigned char), height);

    img = Image_new(width, height, channels,maxval);

    imgData = img_getData(img);

    charIter = charData;
    floatIter = imgData;
    scale = 1.0f / ((float)maxval);

    for (x = 0; x < height; x++) {
        for (y = 0; y < width; y++) {
            for (z = 0; z < channels; z++) {
                *floatIter = ((float)*charIter) * scale;
                floatIter++;
                charIter++;
            }
        }
    }
    return img;
}

void File_write(FILE* file, const void *buffer, size_t size, size_t count) {
    if (file == NULL) {
    	printf("ERROR: File not exist!");
        return ;
    }

    size_t res = fwrite(buffer, size, count, file);
    if (res != count) {
        printf("ERROR: Failed to write PPM file");
        return;
    }
    return ;
}

bool write_image(const char* filename, Image* img) {

    FILE* file;
    int imgWidth, imgHeight, imgChannels;
    int maxval;
    unsigned char* charData;
    unsigned char * charIter;
    float* floatIter;

    file = fopen(filename, "wb+");
    if (file == NULL) {
        printf("Could not open %s in mode %s\n", filename,"wb+");
        return false;
    }
    imgWidth = img_getWidth(img);
    imgHeight = img_getHeight(img);
    imgChannels = img_getChannels(img);
    maxval = img_getMaxval(img);

    if (imgChannels == 3) {
        fprintf(file, "P6\n");
    }
    else
        fprintf(file, "P5\n");

    //fprintf writes the C string pointed by "format" to the "stream"

    fprintf(file, "#Created via PPM!\n");
    fprintf(file, "%d %d\n", imgWidth, imgHeight);
    fprintf(file, "%d\n", maxval);

    charData = (unsigned char*)malloc(sizeof(unsigned char) * imgWidth * imgHeight * imgChannels);

    charIter = charData;
    floatIter = img_getData(img);

    for (int i = 0; i < imgHeight; i++) {
        for (int j = 0; j < imgWidth; j++) {
        	for(int z = 0; z < imgChannels; z++){
				*charIter = (unsigned char)ceil(clamp(*floatIter,0,1) * maxval);
				floatIter++;
				charIter++;
        	}
        }
    }


    File_write(file, charData,imgWidth * imgChannels * sizeof(unsigned char), imgHeight);

    free(charData);
    fflush(file);
    fclose(file);
    return true;
}
