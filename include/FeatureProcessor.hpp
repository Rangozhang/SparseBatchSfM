#ifndef FEATUREPROCESSOR_HPP_
#define FEATUREPROCESSOR_HPP_

#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/core/core.hpp>

#include "protos.hpp"

namespace sparse_batch_sfm {

class FeatureProcessor {
 public:
  explicit FeatureProcessor();
  virtual ~FeatureProcessor();
  // or extractMatch()? put those two in the same function
  bool feature_match(const std::vector<std::unique_ptr<cv::Mat>>& image_seq, FeatureStruct& feature_struct, int minHessian, float match_thres, bool visualize); 
  bool skeletonize(Eigen::Matrix<int, Eigen::Dynamic,
                   Eigen::Dynamic, Eigen::RowMajor>& skeleton, float min_n_matches);
};

} //namespace sparse_batch_sfm

#endif
