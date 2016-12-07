#include "TwoViewReconstruction.hpp"

namespace sparse_batch_sfm {

  bool TwoViewReconstruction::reconstruct(const FeatureStruct& feature_struct, int frame1, int frame2,
                          Eigen::Matrix3d K1, Eigen::Matrix3d K2, GraphStruct& graph) {

    // 1. compute F
    if (!estimateF(feature_struct.feature_point, feature_struct.feature_idx, frame1, frame2)) {
      std::cout << "Failed on ransacF" << std::endl;
      return false;
    }

    // 2. compute E
    E = graph.K[frame1].transpose() * F * graph.K[frame2];

    // 3. Mot from E
    graph.Mot.push_back(RtFromE(graph.K[frame1], graph.K[frame2], feature_struct.feature_point, feature_struct.feature_idx));

    // 4. Triangulation
    if (!triangulate(graph)) {
      std::cout << "Failed on triangulation" << std::endl;
      return false;
    }

    return true;
  }

  bool TwoViewReconstruction::estimateF(const std::vector<FeaturePoint>& feature_point,
                 const Eigen::SparseMatrix<int, Eigen::RowMajor>& feature_idx, int frame1, int frame2) {
    return false;
  }

  Eigen::Matrix<double, 3, 4, Eigen::ColMajor>
                        TwoViewReconstruction::RtFromE(const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2,
                                                       const std::vector<FeaturePoint>& feature_point,
                                                       const Eigen::SparseMatrix<int, Eigen::RowMajor> feature_idx) {
    Eigen::Matrix<double, 3, 4, Eigen::ColMajor> Mot;
    return Mot;
  }

  bool TwoViewReconstruction::triangulate(GraphStruct& graph) {
    return true;
  }

}
