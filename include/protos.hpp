#ifndef PROTOS_HPP_
#define PROTOS_HPP_

#include<Eigen/Dense>
#include<Eigen/Sparse>

struct FeaturePoint {
  Eigen::Vector2d pos;
  Eigen::Vector3i rgb;
}

struct FeatureStruct {
  std::vector<FeaturePoint> feature_point;
  Eigen::SparseMatrix<int, Eigen::RowMajor> feature_idx;
  Eigen::SparseMatrix<int, Eigen::RowMajor> skeleton;
}

struct GraphStruct {
  std::vector<int> frame_idx;
  std::vector<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> K;
  std::vector<Eigen::Matrix<double, 3, 4, Eigen::ColMajor>> Mot;
  Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::ColMajor> Str;
}

#endif
