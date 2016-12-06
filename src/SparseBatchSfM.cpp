#ifndef SPARSEBATCHSFM_CPP_
#define SPARSEBATCHSFM_CPP_

#include "SparseBatchSfM.hpp"

namespace sparse_batch_sfm {

  SparseBatchSfM::SparseBatchSfM() {
    image_capture_.reset(new ImageCapture());
    //feature_processor_.reset(new FeatureProcessor());
  }

  SparseBatchSfM::~SparseBatchSfM() {
    image_capture_.reset(); 
    //feature_processor_.reset();
  }

  SparseBatchSfM* SparseBatchSfM::instance_ = nullptr;

  SparseBatchSfM* SparseBatchSfM::getInstance() {
    if (!instance_) {
      instance_ = new SparseBatchSfM();
    }
    return instance_;
  }

  void SparseBatchSfM::run(const std::string& input_path) {
    std::cout << "INPUT PARAMS" << std::endl;
    std::cout << "Input path: " << input_path << std::endl;
    SparseBatchSfM* controller = controller->getInstance();
    if (!controller->image_capture_->ReadFromDir(
                  input_path, controller->image_seq_)) {
      return;
    }

    int seq_len = controller->image_seq_.size();
    
    controller->feature_processor_->feature_match(controller->image_seq_, controller->feature_struct_, 400, 200, false);
    std::cout << controller->feature_struct_.skeleton << std::endl;
    controller->feature_struct_.skeleton.resize(8, 8);
    controller->feature_struct_.skeleton << 0, 1, 0, 10, 0, 0, 0, 0,
                                            1, 0, 10, 0, 3, 0, 0, 0,
                                            0, 10, 0, 4, 9, 0, 0, 0,
                                            10, 0, 4, 0, 0, 7, 0, 0,
                                            0,  3, 9, 0, 0, 6, 0, 0,
                                            0,  0, 0, 7, 6, 0, 7, 8,
                                            0,  0, 0, 0, 0, 7, 0, 5,
                                            0,  0, 0, 0, 0, 8, 5, 0;
    controller->feature_processor_->skeletonize(controller->feature_struct_.skeleton, 0);
  }

} // namespace sparse_batch_sfm

#endif
