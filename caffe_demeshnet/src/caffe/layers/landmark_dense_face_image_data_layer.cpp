#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/landmark_dense_face_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#define PI 3.1415925

namespace caffe {

template <typename Dtype>
LandmarkDenseFaceImageDataLayer<Dtype>::~LandmarkDenseFaceImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void LandmarkDenseFaceImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.landmark_dense_face_image_data_param().new_height();
  const int new_width  = this->layer_param_.landmark_dense_face_image_data_param().new_width();
  const bool is_color  = this->layer_param_.landmark_dense_face_image_data_param().is_color();
  string root_folder = this->layer_param_.landmark_dense_face_image_data_param().root_folder();
  const int target_size = this->layer_param_.landmark_dense_face_image_data_param().target_size();
  theta_num_ = 6;
  //DIY
  const bool is_color_clean  = this->layer_param_.landmark_dense_face_image_data_param().is_color_clean();
  //const int crop_height = this->layer_param_.landmark_dense_face_image_data_param().crop_height();
  //const int crop_width  = this->layer_param_.landmark_dense_face_image_data_param().crop_width();




  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.landmark_dense_face_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename; 
  string label_filename;
  string clean_filename;
  float eye_l_x, eye_l_y, eye_r_x, eye_r_y;
  while (infile >> filename >> label_filename >> clean_filename >> eye_l_x >> eye_l_y >> eye_r_x >> eye_r_y) {
    lines_.push_back(std::make_pair(filename, label_filename));
    clean_lines_.push_back(std::make_pair(filename, clean_filename));
    left_eyes_.push_back(std::make_pair(eye_l_x, eye_l_y));
    right_eyes_.push_back(std::make_pair(eye_r_x, eye_r_y));
  } 

  if (this->layer_param_.landmark_dense_face_image_data_param().shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " examples.";


  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.landmark_dense_face_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.landmark_dense_face_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);

//DIY for label image
//
  cv::Mat cv_img_clean = ReadImageToCVMat(root_folder + clean_lines_[lines_id_].second,
                                    new_height, new_width, is_color_clean);
  vector<int> top_shape_clean = this->data_transformer_->InferBlobShape(cv_img_clean);
  this->transformed_clean_.Reshape(top_shape_clean);
  // sanity check label image
  cv::Mat cv_lab = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                       new_height, new_width, false, true);
  vector<int> top_shape_label = this->data_transformer_->InferBlobShape(cv_lab);
  this->transformed_label_.Reshape(top_shape_label);
  CHECK(cv_lab.channels() == 1) << "Can only handle grayscale label images";
  CHECK(cv_lab.rows == cv_img.rows && cv_lab.cols == cv_img.cols) << "Input and label "
      << "image heights and widths must match";
  CHECK(cv_lab.rows == cv_img_clean.rows && cv_lab.cols == cv_img_clean.cols) << "Output and label "
      << "image heights and widths must match";



  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.landmark_dense_face_image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top_shape_label[0] = batch_size;
  top[1]->Reshape(top_shape_label);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(top_shape_label);
  }
  // DIY for clean
  top_shape_clean[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].clean_.Reshape(top_shape_clean);
  }
  top[2]->Reshape(top_shape_clean);
  // DIY for theta
  vector<int> top_shape_theta(4,1);
  top_shape_theta[0] = batch_size;
  //top_shape_theta[1] = top_shape_theta[2] = 1;
  top_shape_theta[3] = theta_num_;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_1_.Reshape(top_shape_theta);
  }
  top[3]->Reshape(top_shape_theta);
}


template <typename Dtype>
void LandmarkDenseFaceImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle_four(lines_.begin(), lines_.end(), clean_lines_.begin(), clean_lines_.end(), left_eyes_.begin(), left_eyes_.end(), right_eyes_.begin(), right_eyes_.end(),
              prefetch_rng);
}

template <typename Dtype>
void LandmarkDenseFaceImageDataLayer<Dtype>::ThetaFromLandmark(float eyeLX, float eyeLY, float eyeRX, float eyeRY, int img_size, Blob<Dtype>* dst_theta){
   //cv::Point2f srcTri[3];
   //cv::Point2f dstTri[3];
   //cv::Mat warp_mat(2, 3, CV_32FC1);
   /*  The first fail: Use affine transform
   srcTri[0] = cv::Point2f(32,32);
   srcTri[1] = cv::Point2f(64,32);
   srcTri[2] = cv::Point2f(98,32);
   dstTri[0] = cv::Point2f(eye_l_x, eye_l_y);
   dstTri[2] = cv::Point2f(eye_r_x, eye_r_y);
   dstTri[1] = cv::Point2f((eye_l_x + eye_r_x) / 2, (eye_l_y + eye_r_y) / 2);
   warp_mat = cv::getAffineTransform(srcTri, dstTri);
   //warp_mat = cv::getAffineTransform(dstTri,srcTri);
   */

   /* The second fail: Use rotation and shift
   float angTan = (eyeLY - eyeRY)/(eyeLX - eyeRX);
   float angle = atan(angTan)/PI * 180;
   float dist_dst = sqrt((eyeLX-eyeRX)*(eyeLX-eyeRX)+(eyeLY-eyeRY)*(eyeLY-eyeRY));
   LOG(INFO)<<"dist_dst = "<<dist_dst<<" angle"<<angle;
   float scale = 64.0F/dist_dst;
   cv::Point2f src_center(64, 64);
   cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, scale);
   float shift_x = eyeLX - (rot_mat.at<float>(0,0)*32 + rot_mat.at<float>(0,1)*32 + rot_mat.at<float>(0,2)); 
   float shift_y = eyeLY - (rot_mat.at<float>(1,0)*32 + rot_mat.at<float>(1,1)*32 + rot_mat.at<float>(1,2)); 
   LOG(INFO)<<rot_mat.at<float>(0,0)*64 + rot_mat.at<float>(0,1)*64 + rot_mat.at<float>(0,2);
   rot_mat.at<float>(0,2) = shift_x;
   rot_mat.at<float>(1,2) = shift_y;
   LOG(INFO)<<"shift_x "<<shift_x<<" "<<rot_mat.at<float>(0,2)<<"col and row"<<rot_mat.rows<<" "<<rot_mat.cols;
   */
   /*
   Dtype* theta_data = dst_theta->mutable_cpu_data();
   dst_theta->mutable_cpu_data()[0] = Dtype(rot_mat.at<float>(0,0));
   dst_theta->mutable_cpu_data()[1] = Dtype(rot_mat.at<float>(0,1));
   dst_theta->mutable_cpu_data()[2] = Dtype(rot_mat.at<float>(0,2));
   dst_theta->mutable_cpu_data()[3] = Dtype(rot_mat.at<float>(1,0));
   dst_theta->mutable_cpu_data()[4] = Dtype(rot_mat.at<float>(1,1));
   dst_theta->mutable_cpu_data()[5] = Dtype(rot_mat.at<float>(1,2));
   theta_data[0] = Dtype(rot_mat.at<float>(0,0));
   theta_data[1] = Dtype(rot_mat.at<float>(0,1));
   theta_data[2] = Dtype(rot_mat.at<float>(0,2));
   theta_data[3] = Dtype(rot_mat.at<float>(1,0));
   theta_data[4] = Dtype(rot_mat.at<float>(1,1));
   theta_data[5] = Dtype(rot_mat.at<float>(1,2));

   */

   //float a,b,u,v;
   cv::Mat param_mat(4, 1, CV_32F);
   cv::Mat src_mat(4, 4, CV_32F);
   cv::Mat dst_mat(1, 4, CV_32F);
 
/* 
   src_mat.at<float>(0,0) = (32-89) / 89.0;
   src_mat.at<float>(1,0) = (32-110) / 110.0;
   src_mat.at<float>(2,0) = 1;
   src_mat.at<float>(3,0) = 0;
   src_mat.at<float>(0,1) = (32-110) / 110.0;
   src_mat.at<float>(1,1) = -(32-89) / 89.0;
   src_mat.at<float>(2,1) = 0;
   src_mat.at<float>(3,1) = 1;
   src_mat.at<float>(0,2) = (98-89) / 89.0;
   src_mat.at<float>(1,2) = (32-110) / 110.0;
   src_mat.at<float>(2,2) = 1;
   src_mat.at<float>(3,2) = 0;
   src_mat.at<float>(0,3) = (32-110) / 110.0;
   src_mat.at<float>(1,3) = -(98-89) / 89.0;
   src_mat.at<float>(2,3) = 0;
   src_mat.at<float>(3,3) = 1;
 
   dst_mat.at<float>(0,0) = (eyeLX-89) / 89.0;
   dst_mat.at<float>(0,1) = (eyeLY-110) / 110.0;
   dst_mat.at<float>(0,2) = (eyeRX-89) / 89.0;
   dst_mat.at<float>(0,3) = (eyeRY-110) / 110.0;
*/
/*
   src_mat.at<float>(0,0) = (32-110) / 110.0;
   src_mat.at<float>(1,0) = (32-89) / 89.0;
   src_mat.at<float>(2,0) = 1;
   src_mat.at<float>(3,0) = 0;
   src_mat.at<float>(0,1) = (32-89) / 89.0;
   src_mat.at<float>(1,1) = -(32-110) / 110.0;
   src_mat.at<float>(2,1) = 0;
   src_mat.at<float>(3,1) = 1;
   src_mat.at<float>(0,2) = (32-110) / 110.0;
   src_mat.at<float>(1,2) = (98-89) / 89.0;
   src_mat.at<float>(2,2) = 1;
   src_mat.at<float>(3,2) = 0;
   src_mat.at<float>(0,3) = (98-89) / 89.0;
   src_mat.at<float>(1,3) = -(32-110) / 110.0;
   src_mat.at<float>(2,3) = 0;
   src_mat.at<float>(3,3) = 1;
*/

   src_mat.at<float>(0,0) = (32-64) / 64.0;
   src_mat.at<float>(1,0) = (32-64) / 64.0;
   src_mat.at<float>(2,0) = 1;
   src_mat.at<float>(3,0) = 0;
   src_mat.at<float>(0,1) = (32-64) / 64.0;
   src_mat.at<float>(1,1) = -(32-64) / 64.0;
   src_mat.at<float>(2,1) = 0;
   src_mat.at<float>(3,1) = 1;
   src_mat.at<float>(0,2) = (32-64) / 64.0;
   src_mat.at<float>(1,2) = (98-64) / 64.0;
   src_mat.at<float>(2,2) = 1;
   src_mat.at<float>(3,2) = 0;
   src_mat.at<float>(0,3) = (98-64) / 64.0;
   src_mat.at<float>(1,3) = -(32-64) / 64.0;
   src_mat.at<float>(2,3) = 0;
   src_mat.at<float>(3,3) = 1;

   dst_mat.at<float>(0,0) = (eyeLY-110) / 110.0;
   dst_mat.at<float>(0,1) = (eyeLX-89) / 89.0;
   dst_mat.at<float>(0,2) = (eyeRY-110) / 110.0;
   dst_mat.at<float>(0,3) = (eyeRX-89) / 89.0;

   //LOG(INFO)<<"dst_mat = "<<dst_mat;
   //LOG(INFO)<<"src_mat = "<<src_mat;
  
   /* 
   src_mat.at<float>(0,0) = eyeLX - 89;
   src_mat.at<float>(1,0) = eyeLY - 110;
   src_mat.at<float>(2,0) = 1;
   src_mat.at<float>(3,0) = 0;
   src_mat.at<float>(0,1) = eyeLY-110;
   src_mat.at<float>(1,1) = -1*(eyeLX-89);
   src_mat.at<float>(2,1) = 0;
   src_mat.at<float>(3,1) = 1;
   src_mat.at<float>(0,2) = eyeRX-89;
   src_mat.at<float>(1,2) = eyeRY-110;
   src_mat.at<float>(2,2) = 1;
   src_mat.at<float>(3,2) = 0;
   src_mat.at<float>(0,3) = eyeRY-110;
   src_mat.at<float>(1,3) = -1*(eyeRX-89);
   src_mat.at<float>(2,3) = 0;
   src_mat.at<float>(3,3) = 1;
   
   dst_mat.at<float>(0,0) = 32-64;
   dst_mat.at<float>(0,1) = 32-64;
   dst_mat.at<float>(0,2) = 98-64;
   dst_mat.at<float>(0,3) = 32-64;
   */

   param_mat = dst_mat * ( src_mat.inv() );

   Dtype* theta_data = dst_theta->mutable_cpu_data();
   
   theta_data[0] = Dtype(param_mat.at<float>(0,0));
   theta_data[1] = Dtype(param_mat.at<float>(0,1));
   theta_data[2] = Dtype(param_mat.at<float>(0,2));
   theta_data[3] = Dtype(param_mat.at<float>(0,1)*(-1));
   theta_data[4] = Dtype(param_mat.at<float>(0,0));
   theta_data[5] = Dtype(param_mat.at<float>(0,3));
   
   
   
   //Dtype* theta_data = dst_theta->mutable_cpu_data();
   /*
   theta_data[0] = 2;
   theta_data[1] = 0;
   theta_data[2] = 1.5;
   theta_data[3] = 0;
   theta_data[4] = 2;
   theta_data[5] = 1;
   */
   //LOG(INFO)<<"theta and at "<<theta_data[0]<<" "<<Dtype(param_mat.at<float>(0,0));
   //LOG(INFO)<<"at "<<Dtype(param_mat.at<float>(0,0))<<" "<<Dtype(param_mat.at<float>(0,1))<<" "<<Dtype(param_mat.at<float>(0,2));
   //LOG(INFO)<<"param_mat "<<param_mat;
   //cv::Mat temp(3,1, CV_32F);
   //temp.at<float>(0,0) = 32;
   //temp.at<float>(0,1) = 32;
   //temp.at<float>(0,2) = 1;

   //LOG(INFO)<<param_mat.at<float>(0,0)*32 + param_mat.at<float>(0,1)*32 + param_mat.at<float>(0,2)<<" GT is "<<"[ "<<eyeLX<<", "<<eyeLY<<" ]";
   //LOG(INFO)<<theta_data[3]*32 + theta_data[4]*32 + theta_data[5]<<" GT is "<<"[ "<<eyeLX<<", "<<eyeLY<<" ]";
   //LOG(INFO)<<*theta_data;

}

  //

// This function is called on prefetch thread
template <typename Dtype>
void LandmarkDenseFaceImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->clean_.count());
  CHECK(this->transformed_data_.count());
  LandmarkDenseFaceImageDataParameter landmark_dense_face_image_data_param = this->layer_param_.landmark_dense_face_image_data_param();
  const int batch_size = landmark_dense_face_image_data_param.batch_size();
  const int new_height = landmark_dense_face_image_data_param.new_height();
  const int new_width = landmark_dense_face_image_data_param.new_width();
  const bool is_color = landmark_dense_face_image_data_param.is_color();
  string root_folder = landmark_dense_face_image_data_param.root_folder();
  const int target_size = landmark_dense_face_image_data_param.target_size();  
  const bool is_color_clean = landmark_dense_face_image_data_param.is_color_clean(); 

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

//DIY for label and clean
//
  cv::Mat cv_lab = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
      new_height, new_width, false);
  CHECK(cv_lab.data) << "Could not load " << lines_[lines_id_].second;
  vector<int> top_shape_label = this->data_transformer_->InferBlobShape(cv_lab);
  this->transformed_label_.Reshape(top_shape_label);
  top_shape_label[0] = batch_size;
  batch->label_.Reshape(top_shape_label);

  cv::Mat cv_img_clean = ReadImageToCVMat(root_folder + clean_lines_[lines_id_].second,
      new_height, new_width, is_color_clean);
  CHECK(cv_img_clean.data) << "Could not load " << clean_lines_[lines_id_].second;
  vector<int> top_shape_clean = this->data_transformer_->InferBlobShape(cv_img_clean);
  this->transformed_clean_.Reshape(top_shape_clean);
  top_shape_clean[0] = batch_size;
  batch->clean_.Reshape(top_shape_clean);

  vector<int> top_shape_theta(4,1);
  top_shape_theta[3] = theta_num_;
  this->transformed_data_1_.Reshape(top_shape_theta);  
  top_shape_theta[0] = batch_size;
  batch->data_1_.Reshape(top_shape_theta);



  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_clean = batch->clean_.mutable_cpu_data();
  Dtype* prefetch_data_1 = batch->data_1_.mutable_cpu_data();
  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    //int temp;
    //temp = cv_img.at<uchar>(100,100); 
    //LOG(INFO) << "cv_img at <100,100> is " << temp;
    cv::Mat cv_img_clean = ReadImageToCVMat(root_folder + clean_lines_[lines_id_].second,
        new_height, new_width, is_color_clean);
    //temp = cv_img_clean.at<uchar>(100,100);
    //LOG(INFO) << "cv_img_clean at <100,100> is " << temp;
    CHECK(cv_img_clean.data) << "Could not load " << clean_lines_[lines_id_].second;
    cv::Mat cv_lab = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
        new_height, new_width, false);
    CHECK(cv_lab.data) << "Could not load " << lines_[lines_id_].second;
    if (this->layer_param_.landmark_dense_face_image_data_param().mirror()) {
      const bool do_mirror = caffe_rng_rand() % 2;
      if (do_mirror) {
        cv::flip(cv_img,cv_img,1);
        cv::flip(cv_lab,cv_lab,1);
        cv::flip(cv_img_clean,cv_img_clean,1);
      }
    }

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    this->transformed_clean_.set_cpu_data(prefetch_clean + offset);
    this->data_transformer_->Transform(cv_img_clean, &(this->transformed_clean_));
    this->transformed_label_.set_cpu_data(prefetch_label + offset);
    this->data_transformer_->Transform(cv_lab, &(this->transformed_label_),true);
    trans_time += timer.MicroSeconds();
    //LOG(INFO) << "cv_img_clean at <100,100> is " << this->transformed_label_;
    //prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    int offset_theta = batch->data_1_.offset(item_id); 
    this->transformed_data_1_.set_cpu_data(prefetch_data_1 + offset_theta);       
    //Dtype* theta_temp = this->transformed_data_1.mutable_cpu_data();
    //float* theta_from_landmark;
    //ThetaFromLandmark(left_eyes_[lines_id_].first, left_eyes_[lines_id].second, right_eyes_[lines_id].first, right_eyes_[lines_id].second, target_size, theta_from_landmark);
    ThetaFromLandmark(left_eyes_[lines_id_].first, left_eyes_[lines_id_].second, right_eyes_[lines_id_].first, right_eyes_[lines_id_].second, target_size, &(this->transformed_data_1_));
    //LOG(INFO)<<"clean_label "<<this->transformed_clean_.cpu_data()[0];   

    //LOG(INFO)<<"data_1_ "<<this->transformed_data_1_.cpu_data()[0]<<" "<<this->transformed_data_1_.cpu_data()[1]<<" "<<this->transformed_data_1_.cpu_data()[2]<< " "<<this->transformed_data_1_.cpu_data()[3]<< " "<<this->transformed_data_1_.cpu_data()[4]<<" "<<this->transformed_data_1_.cpu_data()[5];
    //for (int i = 0; i<theta_num_,i++){
    //    theta_temp[i] = Dtype(theta_from_landmark[i]);
    //}


    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.landmark_dense_face_image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(LandmarkDenseFaceImageDataLayer);
REGISTER_LAYER_CLASS(LandmarkDenseFaceImageData);

}  // namespace caffe
#endif  // USE_OPENCV
