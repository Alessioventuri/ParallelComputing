#ifndef FUNCTIONIMAGE_H_
#define FUNCTIONIMAGE_H_

#include "Image.h"

Image* import_PPM(const char* filename);
bool write_image(const char* filename, Image* img);

static inline float clamp(float x, float start, float end) {
    float max = x > start ? x : start;
    float min = max < end ? max : end;
    return min;
}

#endif /* FUNCTIONIMAGE_H_ */
