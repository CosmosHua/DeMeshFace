#ifndef CAFFE_FOUR_DENSE_FACE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_FOUR_DENSE_FACE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
namespace caffe {




template <typename Dtype>
class FourDenseFaceImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit FourDenseFaceImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~FourDenseFaceImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FourDenseFaceImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 4; }
  
 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, std::string> > lines_;
  vector<std::pair<std::string, std::string> > clean_lines_;
  int Rand(int n);
  int lines_id_;
  Blob<Dtype> transformed_clean_;
  Blob<Dtype> transformed_label_;
  Blob<Dtype> transformed_data_1_;
  shared_ptr<Caffe::RNG> rng_;
};

}

#endif
