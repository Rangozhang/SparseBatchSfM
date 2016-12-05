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
//   bool extract(const std::unique_ptr<std::unique_ptr<cv::Mat>>& image_seq>, rawSIFT); 
//   bool match(rawSIFT, FeatureStruct& feature_struct);
   bool skeletonize(Eigen::SparseMatrix<int, Eigen::RowMajor>& skeleton);
};

} //namespace sparse_batch_sfm

#endif
