#ifndef TWOVIEWRECONSTRUCTION_HPP_
#define TWOVIEWRECONSTRUCTION_HPP_

#include <iostream>
#include <vector>
#include <memory>

#include "protos.hpp"

/*
 K
*/

namespace sparse_batch_sfm {

class TwoViewReconstruction {
 private:
  Eigen::Matrix<double, 3, 3, Eigen::ColMajor> F;
  Eigen::Matrix<double, 3, 3, Eigen::ColMajor> E;

  bool estimateF(const std::vector<FeaturePoint>& feature_point,
                               const Eigen::SparseMatrix<int, Eigen::RowMajor> feature_idx);
  Eigen::Matrix<double, 3, 4, Eigen::ColMajor>> RtFromE(const Matrix3d& K1, const Matrix3d& K2,
                                                        const std::vector<FeaturePoint>& feature_point,
                                                        const Eigen::SparseMatrix<int, Eigen::RowMajor> feature_idx);
  bool triangulate(GraphStruct& graph);
 public:
  GraphStruct reconstruct(const FeatureStruct& feature_struct, int frame1, int frame2);
};

} // namespace sparse_batch_sfm

#endif
