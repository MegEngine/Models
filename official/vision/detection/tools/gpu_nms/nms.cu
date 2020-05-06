#include "megbrain_pubapi.h"
#include <iostream>
#include <vector>
#include <assert.h>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << " " << cudaGetErrorString(error); \
    } \
  } while (0)
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

int const threadsPerBlock = sizeof(unsigned long long) * 8; // 64

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

template <int unroll = 4>
static inline void cpu_unroll_for(unsigned long long *dst, const unsigned long long *src, int n) {
    int nr_out = (n - n % unroll) / unroll;
    for (int i = 0; i < nr_out; ++i) {
#pragma unroll
        for (int j = 0; j < unroll; ++j) {
            *(dst++) |= *(src++);
        }
    }
    for (int j = 0; j < n % unroll; ++j) {
        *(dst++) |= *(src++);
    }
}

using std::vector;
// const int nr_init_box = 8000;
// vector<unsigned long long> _mask_host(nr_init_box * (nr_init_box / threadsPerBlock));
// vector<unsigned long long> _remv(nr_init_box / threadsPerBlock);
// vector<int> _keep_out(nr_init_box);

// NOTE: If we directly use this lib in nmp.py, we will meet the same _mask_host and other 
// objects, which is not safe for multi-processing programs.

class HostDevice{
protected:
    static const int nr_init_box = 8000;
public:
    vector<unsigned long long> mask_host;
    vector<unsigned long long> remv;
    vector<int> keep_out;

    HostDevice(): mask_host(nr_init_box * (nr_init_box / threadsPerBlock)), remv(nr_init_box / threadsPerBlock), keep_out(nr_init_box){}
};

extern "C"{
    using MGBDevTensor = mgb::pubapi::DeviceTensor;
    using std::cout;

    void * CreateHostDevice(){
        return new HostDevice();
    }
   
    int NMSForwardGpu(void* box_ptr, void* mask_ptr, void* output_ptr, void* output_num_ptr, float iou_threshold, int max_output, void* host_device_ptr){
        auto box_tensor = mgb::pubapi::as_versioned_obj<MGBDevTensor>(box_ptr);
        auto mask_tensor= mgb::pubapi::as_versioned_obj<MGBDevTensor>(mask_ptr);
        auto output_tensor = mgb::pubapi::as_versioned_obj<MGBDevTensor>(output_ptr);
        auto output_num_tensor = mgb::pubapi::as_versioned_obj<MGBDevTensor>(output_num_ptr);

        // auto cuda_stream = static_cast<cudaStream_t> (box_tensor->desc.cuda_ctx.stream);
         auto cuda_stream = static_cast<cudaStream_t> (output_tensor->desc.cuda_ctx.stream);
        // assert(box_tensor->desc.shape[0] == output_tensor->desc.shape[0]);

        // cout << "box_tensor.ndim: " << box_tensor->desc.ndim << "\n";
        // cout << "box_tensor.shape_0: " << box_tensor->desc.shape[0] << "\n";
        // cout << "box_tensor.shape_1: " << box_tensor->desc.shape[1] << "\n";
        int box_num = box_tensor->desc.shape[0];
        int box_dim = box_tensor->desc.shape[1];
        assert(box_dim == 5);

        const int col_blocks = DIVUP(box_num, threadsPerBlock);
        // cout << "mask_dev size: " << box_num * col_blocks * sizeof(unsigned long long) << "\n";
        // cout << "mask_ptr size: " << mask_tensor->desc.shape[0] * sizeof(int) << "\n";
        // cout << "mask shape : " << mask_tensor->desc.shape[0] << "\n";

        dim3 blocks(DIVUP(box_num, threadsPerBlock), DIVUP(box_num, threadsPerBlock));
        // dim3 blocks(col_blocks, col_blocks);
        dim3 threads(threadsPerBlock);
        // cout << "sizeof unsigned long long " << sizeof(unsigned long long) << "\n"; 
        float* dev_box = static_cast<float*> (box_tensor->desc.dev_ptr);
        unsigned long long* dev_mask = static_cast<unsigned long long*> (mask_tensor->desc.dev_ptr);
        int * dev_output = static_cast<int*> (output_tensor->desc.dev_ptr);

        CUDA_CHECK(cudaMemsetAsync(dev_mask, 0, mask_tensor->desc.shape[0] * sizeof(int), cuda_stream));
        // CUDA_CHECK(cudaMemsetAsync(dev_output, 0, output_tensor->desc.shape[0] * sizeof(int), cuda_stream));
        nms_kernel<<<blocks, threads, 0, cuda_stream>>>(box_num, iou_threshold, dev_box, dev_mask);
        // cudaDeviceSynchronize();
        
        // get the host device vectors
        HostDevice* host_device =  static_cast<HostDevice* >(host_device_ptr);
        vector<unsigned long long>& _mask_host = host_device->mask_host;
        vector<unsigned long long>& _remv      = host_device->remv;
        vector<int>& _keep_out                 = host_device->keep_out;


        int current_mask_host_size = box_num * col_blocks;
        if(_mask_host.capacity() < current_mask_host_size){
            _mask_host.reserve(current_mask_host_size);
        }
        CUDA_CHECK(cudaMemcpyAsync(&_mask_host[0], dev_mask, sizeof(unsigned long long) * box_num * col_blocks, cudaMemcpyDeviceToHost, cuda_stream));
        // cout << "\n m_host site: " << static_cast<void *> (&_mask_host[0]) << "\n";

        if(_remv.capacity() < col_blocks){
            _remv.reserve(col_blocks);
        }
        if(_keep_out.capacity() < box_num){
            _keep_out.reserve(box_num);
        }
        if(max_output < 0){
            max_output = box_num;
        }
        memset(&_remv[0], 0, sizeof(unsigned long long) * col_blocks);
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

        // do the cpu reduce
        int num_to_keep = 0;
        for (int i = 0; i < box_num; i++) {
            int nblock = i / threadsPerBlock;
            int inblock = i % threadsPerBlock;

            if (!(_remv[nblock] & (1ULL << inblock))) {
                _keep_out[num_to_keep++] = i;
                if(num_to_keep == max_output){
                    break;
                }
                // NOTE: here we need add nblock to pointer p
                unsigned long long *p = &_mask_host[0] + i * col_blocks + nblock;
                unsigned long long *q = &_remv[0] + nblock;
                cpu_unroll_for(q, p, col_blocks - nblock);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(dev_output, &_keep_out[0], num_to_keep * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));
        int* dev_output_num = static_cast<int*>(output_num_tensor->desc.dev_ptr);
        CUDA_CHECK(cudaMemcpyAsync(dev_output_num, &num_to_keep, sizeof(int), cudaMemcpyHostToDevice, cuda_stream));
        // CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        return num_to_keep;
    }
}
