#ifndef SPARSEBATCHSFM_HPP_
#define SPARSEBATCHSFM_HPP_

#include <memory>
#include <string>

#include "ImageCapture.hpp"
#include "FeatureProcessor.hpp"
#include "TwoViewReconstruction.hpp"
#include "utils.hpp"

namespace sparse_batch_sfm {

class SparseBatchSfM {
 public:
  explicit SparseBatchSfM();
  virtual ~SparseBatchSfM();

  static SparseBatchSfM* getInstance();
  static void run(const std::string& input_path);
  bool writeGraphToPLYFile(const std::vector<std::unique_ptr<GraphStruct>>& graphs,
                                  const char* filename);

 private:
  static SparseBatchSfM* instance_;

  // input images
  std::unique_ptr<ImageCapture> image_capture_;
  std::unique_ptr<FeatureProcessor> feature_processor_;
  std::unique_ptr<TwoViewReconstruction> twoview_reconstruction_;

 private:
  // input video sequence
  std::vector<std::unique_ptr<cv::Mat>> image_seq_;
  FeatureStruct feature_struct_;
  std::vector<std::unique_ptr<GraphStruct>> graphs_;
};

} // namespace sparse_batch_sfm

#endif
