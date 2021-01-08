#ifndef IMAGE_H_
#define IMAGE_H_

typedef struct {
    int width;
    int height;
    int channels;
    int maxval;
    float* data;

} Image;

#define img_channels 3

inline int img_getChannels(Image* img) {return img->channels;}
inline int img_getHeight(Image* img) {return img->height;}
inline int img_getMaxval(Image* img) {return img->maxval;}
inline int img_getWidth(Image* img) {return img->width;}
inline void img_setWidth(Image* img,int width) { img->width = width;}

inline float* img_getData(Image* img) {return img->data;}
inline void img_setChannels(Image* img,int channels) {img->channels = channels;}
inline void img_setData(Image* img,float *data) {img->data = data;}
inline void img_setHeight(Image* img,int height) {img->height = height;}
inline void img_setMaxval(Image* img,int maxval) {img->maxval = maxval;}

Image* Image_new(int width, int height, int channels, float* data,int maxval);
Image* Image_new(int width, int height, int channels,int maxval);
Image* Image_new(int width, int height, int maxval);

void Image_delete(Image* img);


#endif /* IMAGE_H_ */
