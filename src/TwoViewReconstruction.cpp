#include "TwoViewReconstruction"

namespace sparse_batch_sfm {

  bool reconstruct(const FeatureStruct& feature_struct, int frame1, int frame2,
                          Eigen::Matrix3d K1, Eigen::Matrix3d K2, GraphStruct& graph) {

    // 1. compute F
    if (!estimateF(feature_struct.feature_point, feature_struct.feature_idx)) {
      std::cout << "Failed on ransacF" << std::endl;
      return false;
    }

    // 2. compute E
    E = graph.K[frame1].transpose() * F * graph.K[frame2];

    // 3. Mot from E
    Mot.push_back(RtFromE(graph.K[frame1], graph.K[frame2], feature_struct.feature_point, feature_struct.feature_idx));

    // 4. Triangulation
    if (!triangulate(graph)) {
      std::cout << "Failed on triangulation" << std::endl;
      return false;
    }

    return true;
  }

}
