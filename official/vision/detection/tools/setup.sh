nvcc -I /usr/local/lib/python3.6/dist-packages/megengine/_internal/include -shared -o lib_nms.so -Xcompiler "-fno-strict-aliasing -fPIC" ./gpu_nms/nms.cu
