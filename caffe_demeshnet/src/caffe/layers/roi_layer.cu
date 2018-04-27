#include <vector>

#include "caffe/layers/roi_layer.hpp"

namespace caffe {

template <typename Dtype>
void RoiLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   Forward_gpu(bottom, top);
}

template <typename Dtype>
void RoiLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(RoiLayer);


}  // namespace caffe
