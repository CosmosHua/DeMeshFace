#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layers/berhu_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BerhuLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  max_c_ = 0;
}

template <typename Dtype>
void BerhuLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  
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
  //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BerhuLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BerhuLossLayer);
#endif

INSTANTIATE_CLASS(BerhuLossLayer);
REGISTER_LAYER_CLASS(BerhuLoss);

}  // namespace caffe
