#include <vector>

#include "caffe/layers/roi_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RoiEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  diff_weight_.ReshapeLike(*bottom[0]);
  roi_mat_.ReshapeLike(*bottom[0]);
  int num = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  if (bottom[0]->width() == 85) {
      for (int i = 0; i < num; i++){
      for (int j = 0; j < channel; j++){  
      for (int h = 0; h < height; h++){
      for (int w = 0; w < width; w++){
           if ((h<88) && (h >=27) && (w<72) && (w>=11)){
               roi_mat_.mutable_cpu_data()[(((i*channel+j)*height+h)*width + w)] = Dtype(1);
           }else{ 
               roi_mat_.mutable_cpu_data()[(((i*channel+j)*height+h)*width + w)] = Dtype(0);
           }
          }
        }
      }
   }
  }
  if (bottom[0]->width() == 41) {
      for (int i = 0; i < num; i++){
      for (int j = 0; j < channel; j++){
      for (int h = 0; h < height; h++){
      for (int w = 0; w < width; w++){
           if ((h<42) && (h >=13) && (w<33) && (w>=5)){
               roi_mat_.mutable_cpu_data()[(((i*channel+j)*height+h)*width + w)] = Dtype(1);
           }else{
               roi_mat_.mutable_cpu_data()[(((i*channel+j)*height+h)*width + w)] = Dtype(0);
           }
          }
        }
      }
   }


  }
}

template <typename Dtype>
void RoiEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_mul(count, diff_.cpu_data(), roi_mat_.cpu_data(), diff_weight_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_weight_.cpu_data(), diff_weight_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void RoiEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_weight_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RoiEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(RoiEuclideanLossLayer);
REGISTER_LAYER_CLASS(RoiEuclideanLoss);

}  // namespace caffe
