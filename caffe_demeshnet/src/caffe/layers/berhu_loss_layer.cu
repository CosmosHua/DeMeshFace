#include <vector>

#include "caffe/layers/berhu_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BerhuLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  for (int i = 0; i < count; i++){
      if (fabs(diff_.cpu_data()[i]) > max_c_){
                max_c_ = fabs(diff_.cpu_data()[i]);}
  }
  max_c_ = 0.2 * max_c_;
  Dtype loss = Dtype(0);
  Dtype *diff = diff_.mutable_cpu_data();
  for (int i = 0; i < count; i++){
      if (fabs(diff[i]) <= max_c_){
          loss += fabs(diff[i]);
      }else{
          loss += (diff[i]*diff[i] + max_c_*max_c_) / (Dtype(2)*max_c_);
      }
      if (diff[i] <= Dtype(1) && diff[i] > Dtype(0)){
          diff[i] = Dtype(1);
      }else if (diff[i] == Dtype(0)){
          diff[i] = Dtype(0);
      }else if (diff[i] > Dtype(-1) && diff[i] < Dtype(0)){
          diff[i] = Dtype(-1);
      }else{
          diff[i] = diff[i] / max_c_;
      }
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BerhuLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BerhuLossLayer);

}  // namespace caffe
