#ifndef IMAGECAPTURE_HPP_
#define IMAGECAPTURE_HPP_

#include <string.h>
#include <memory>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "utils.hpp"

namespace sparse_batch_sfm {

class ImageCapture {
 public:
  bool ReadFromDir(const std::string& input_path,
          std::vector<std::unique_ptr<cv::Mat>>& image_seq);
};

} // namespace sparse_batch_sfm

#endif
