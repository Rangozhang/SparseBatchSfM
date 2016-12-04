#ifndef SPARSEBATCHSFM_CPP_
#define SPARSEBATCHSFM_CPP_

#include "SparseBatchSfM.hpp"

namespace sparse_batch_sfm {

  SparseBatchSfM::SparseBatchSfM() {
    image_capture_.reset(new ImageCapture());
  }

  SparseBatchSfM::~SparseBatchSfM() {
    image_capture_.reset(); 
  }

  SparseBatchSfM* SparseBatchSfM::instance_ = nullptr;

  SparseBatchSfM* SparseBatchSfM::getInstance() {
    if (!instance_) {
      instance_ = new SparseBatchSfM();
    }
    return instance_;
  }

  void SparseBatchSfM::run(const string& input_path) {
    cout << "INPUT PARAMS" << endl;
    cout << "Input path: " << input_path << endl;
    SparseBatchSfM* controller = controller->getInstance();
    if (!controller->image_capture_->ReadFromDir(
                  input_path, controller->image_seq)) {
      return;
    }

    int seq_len = controller->image_seq.size();

  }

} // namespace sparse_batch_sfm

#endif
