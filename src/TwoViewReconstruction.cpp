#include "TwoViewReconstruction"

namespace sparse_batch_sfm {

  GraphStruct reconstruct(const FeatureStruct& feature_struct, int frame1, int frame2) {
    GraphStruct graph;
    // TODO: change to calibrated K
    graph.K << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    // 1. compute F
    estimateF(feature_struct.feature_point, feature_struct.feature_idx);
    // 2. compute E
    E = graph.K[frame1].transpose() * F * graph.K[frame2];
    // 3. Mot from E
    Mot.push_back(RtFromE())
  }

}
