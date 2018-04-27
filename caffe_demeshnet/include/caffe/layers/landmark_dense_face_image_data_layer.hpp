#ifndef CAFFE_LANDMARK_DENSE_FACE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_LANDMARK_DENSE_FACE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {




template <typename Dtype>
class LandmarkDenseFaceImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit LandmarkDenseFaceImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~LandmarkDenseFaceImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseFaceImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  void ThetaFromLandmark(float eye_l_x, float eye_l_y, float eye_r_x, float eye_r_y, int img_size, Blob<Dtype>* dst_theta);
  
  vector<std::pair<std::string, std::string> > lines_;
  vector<std::pair<std::string, std::string> > clean_lines_;
  vector<std::pair<float, float> > left_eyes_; 
  vector<std::pair<float, float> > right_eyes_; 
  int lines_id_;
  
  Blob<Dtype> transformed_clean_;
  Blob<Dtype> transformed_label_;
  Blob<Dtype> transformed_data_1_;
  int theta_num_;
};

}

#endif
