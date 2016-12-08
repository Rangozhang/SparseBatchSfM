#ifndef TWOVIEWRECONSTRUCTION_HPP_
#define TWOVIEWRECONSTRUCTION_HPP_

#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>

#include "protos.hpp"

/*
 K
*/

namespace sparse_batch_sfm {

class TwoViewReconstruction {
 private:
  Eigen::Matrix3d F_;
  Eigen::Matrix3d E_;
  
  void printE() {
    std::cout << "Essential Matrix: " << std::endl
        << E_ << std::endl << std::endl;
  };

  void printF() {
    std::cout << "Fundamental Matrix: " << std::endl
        << F_ << std::endl << std::endl;
  };

  bool estimateF(FeatureStruct& feature_struct,
                 int frame1, int frame2,
                 int img_width, int img_height, const cv::Mat& img1, const cv::Mat& img2);

  Eigen::Matrix<double, 3, 4, Eigen::ColMajor> RtFromE(const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2,
                                                       const FeatureStruct& feature_struct,
                                                       int frame1, int frame2);

  bool triangulate(const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2,
                   const Eigen::Matrix<double, 3, 4>& Mot, const FeatureStruct& feature_struct,
                   int frame1, int frame2, Eigen::Matrix<double, 6, Eigen::Dynamic>& Str);

 public:
  bool reconstruct(FeatureStruct& feature_struct, int frame1, int frame2, int img_width, int img_height,
                   Eigen::Matrix3d K1, Eigen::Matrix3d K2, GraphStruct& graph, const cv::Mat& img1, const cv::Mat& img2);
};

} // namespace sparse_batch_sfm

#endif
