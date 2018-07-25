# TensorRT-SSD-facedet
This code is sample to implement caffe-ssd.

This sample implements a fast face detector based on caffe-ssd framework.

This model can run more than 100fps on gtx1080ti!

The code has been tested by myself, it can help you learn about TensorRT API and ssd fast!

## requirmets:
1.TensorRT 4.0.1
2.Cuda8.0 / Cuda9.0 and Cudnn 7.1
3.OpenCV

## about TensorRT Scale layer
Scale layer has been supported officially, but you must provide both scale param and bias param!!!

If you just provide scale params, it won't work!

## about TensorRT Eltwise layer
Scale layer has been supported officially too, but you can only provide two blobs as input!!!
