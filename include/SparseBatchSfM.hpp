#ifndef SPARSEBATCHSFM_HPP_
#define SPARSEBATCHSFM_HPP_

#include <memory>
#include <string>

#include "ImageCapture.hpp"
#include "utils.hpp"

namespace sparse_batch_sfm {

class SparseBatchSfM {
 public:
  explicit SparseBatchSfM();
  virtual ~SparseBatchSfM();

  static SparseBatchSfM* getInstance();
  static void run(const string& input_path);

 private:
  static SparseBatchSfM* instance_;

  // input images
  std::unique_ptr<ImageCapture> image_capture_;

 private:
  // input video sequence
  vector<unique_ptr<Mat>> image_seq;
};

} // namespace sparse_batch_sfm

#endif
