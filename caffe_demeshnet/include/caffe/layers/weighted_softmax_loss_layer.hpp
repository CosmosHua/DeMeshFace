#ifndef CAFFE_WEIGHTED_SOFTMAX_WITH_LOSS_LAYER_HPP_
#define CAFFE_WEIGHTED_SOFTMAX_WITH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {




template <typename Dtype>
class WeightedSoftmaxWithLossLayer : public LossLayer<Dtype> {
 public:
   /**
 *     * @param param provides LossParameter loss_param, with options:
 *         *  - ignore_label (optional)
 *             *    Specify a label value that should be ignored when computing the loss.
 *                 *  - normalize (optional, default true)
 *                     *    If true, the loss is normalized by the number of (nonignored) labels
 *                         *    present; otherwise the loss is simply summed over spatial locations.
 *                             */
  explicit WeightedSoftmaxWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxWithLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /**
 *    * @brief Computes the softmax loss error gradient w.r.t. the predictions.
 *       *
 *          * Gradients cannot be computed with respect to the label inputs (bottom[1]),
 *             * so this method ignores bottom[1] and requires !propagate_down[1], crashing
 *                * if propagate_down[1] is set.
 *                   *
 *                      * @param top output Blob vector (length 1), providing the error gradient with
 *                         *      respect to the outputs
 *                            *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *                               *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
 *                                  *      as @f$ \lambda @f$ is the coefficient of this layer's output
 *                                     *      @f$\ell_i@f$ in the overall Net loss
 *                                        *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
 *                                           *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
 *                                              *      (*Assuming that this top Blob is not used as a bottom (input) by any
 *                                                 *      other layer of the Net.)
 *                                                    * @param propagate_down see Layer::Backward.
 *                                                       *      propagate_down[1] must be false as we can't compute gradients with
 *                                                          *      respect to the labels.
 *                                                             * @param bottom input Blob vector (length 2)
 *                                                                *   -# @f$ (N \times C \times H \times W) @f$
 *                                                                   *      the predictions @f$ x @f$; Backward computes diff
 *                                                                      *      @f$ \frac{\partial E}{\partial x} @f$
 *                                                                         *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *                                                                            *      the labels -- ignored as we can't compute their error gradients
 *                                                                               */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  shared_ptr<Layer<Dtype> > softmax_layer_;
  Blob<Dtype> prob_;
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
  bool has_ignore_label_;
  int ignore_label_;
  bool normalize_;
  bool weight_by_label_freqs_;
  Blob<float> label_counts_;

  int softmax_axis_, outer_num_, inner_num_;
};
}

#endif 