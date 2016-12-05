#include <unordered_set>

#include "FeatureProcessor.hpp"


namespace sparse_batch_sfm {
  bool FeatureProcessor::feature_match(const std::vector<std::unique_ptr<cv::Mat>>& image_seq, FeatureStruct& feature_struct) {
  
  return true;
  }

  bool FeatureProcessor::skeletonize(Eigen::SparseMatrix<int, Eigen::RowMajor>& skeleton) {
    
  }
}
