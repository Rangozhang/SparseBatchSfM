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
  
  void printE() {
    std::cout << "Essential Matrix: " << std::endl
        << E << std::endl << std::endl;
  };

  void printF() {
    std::cout << "Fundamental Matrix: " << std::endl
        << F << std::endl << std::endl;
  };

  bool estimateF(const std::vector<FeaturePoint>& feature_point,
                 const Eigen::SparseMatrix<int, Eigen::RowMajor>& feature_idx, int frame1, int frame2);
  Eigen::Matrix<double, 3, 4, Eigen::ColMajor> RtFromE(const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2,
                                                        const std::vector<FeaturePoint>& feature_point,
                                                        const Eigen::SparseMatrix<int, Eigen::RowMajor> feature_idx);
  bool triangulate(GraphStruct& graph);
 public:
  bool reconstruct(const FeatureStruct& feature_struct, int frame1, int frame2,
                   Eigen::Matrix3d K1, Eigen::Matrix3d K2, GraphStruct& graph);
};

} // namespace sparse_batch_sfm

#endif
