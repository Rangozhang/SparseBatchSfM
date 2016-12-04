#ifndef IMAGECAPTURE_HPP_
#define IMAGECAPTURE_HPP_

#include <string.h>
#include <memory>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "utils.hpp"

using namespace cv;
using namespace std;

namespace sparse_batch_sfm {

class ImageCapture {
 public:
  bool ReadFromDir(const string& input_path,
          vector<unique_ptr<Mat>>& image_seq);
};

} // namespace sparse_batch_sfm

#endif
