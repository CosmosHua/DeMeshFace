#include <algorithm>
#include <vector>

#include "caffe/layers/roi_layer.hpp"

namespace caffe {

template <typename Dtype>
void RoiLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
    N_ = bottom[0]->num();
    C_ = bottom[0]->channels();
    W_ = bottom[0]->width();
    H_ = bottom[0]->height();
    width_ = this->layer_param_.roi_param().new_width();
    height_ = this->layer_param_.roi_param().new_height();
}

template <typename Dtype> 
void RoiLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
    vector<int> top_shape(4,1);
    top_shape[0] = N_;
    top_shape[1] = C_;
    top_shape[2] = height_;
    top_shape[3] = width_;
    top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RoiLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
 
  for (int i = 0; i < N_; ++i) {
     for (int j = 0; j < C_; ++j){
         for (int m = 0; m < height_; ++m){
             for (int n = 0; n< width_; ++n){
                   top_data[((i*C_ + j)*height_ + m )*width_ +n] = bottom_data[((i*C_ + j)*H_ + m )*W_ +n];
      }
   }
  }
}
}

template <typename Dtype>
void RoiLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
      bottom_diff[i] = 0;
    }

    for (int i = 0; i < N_; ++i) {
       for (int j = 0; j < C_; ++j){
          for (int m = 0; m < height_; ++m){
             for (int n = 0; n< width_; ++n){
                bottom_diff[((i*C_ + j)*H_ + m )*W_ +n] = top_diff[((i*C_ + j)*height_ + m )*width_ +n];
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(RoiLayer);
#endif

INSTANTIATE_CLASS(RoiLayer);

}  // namespace caffe
