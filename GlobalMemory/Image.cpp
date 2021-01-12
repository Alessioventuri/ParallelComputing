#include "Image.h"

#include <iostream>
#include <cassert>
#include <cstdlib>

Image* Image_new(int width, int height, int channels, float* data,int maxval) {
    Image* img;

    img = (Image*)malloc(sizeof(Image));

    img_setWidth(img, width);
    img_setHeight(img, height);
    img_setChannels(img, channels);
    img_setMaxval(img, maxval);

    img_setData(img, data);
    return img;
}

Image* Image_new(int width, int height, int channels,int maxval) {
    float* data = (float*)malloc(sizeof(float) * width * height * channels);
    return Image_new(width, height, channels, data,maxval);
}

Image* Image_new(int width, int height,int maxval) {
    return Image_new(width, height, img_channels,maxval);
}

void Image_delete(Image* img) {
    if (img != NULL) {
        if (img_getData(img) != NULL) {
            free(img_getData(img));
        }
        free(img);
    }
}


