#include "GraphMerge.hpp"

namespace sparse_batch_sfm {

    bool GraphMerge::merge(GraphStruct &graphA, const GraphStruct &graphB) {
      // find common frame and new frame
      if (!findCommonFrame(graphA.frames, graphB.frames)) {
        return false;
      }

      // transform graphB into the world coordinate of graphA


      // update the merged graph
      int num_point = graphB.feature_points.size();
      for (int i = 0; i < num_point; i++) {
        std::cout << graphB.SparseMatrix(commonFrameIdx_, i) << std::endl;
      }

      for (int i = 0; i < frames1.size(); i++) {
        std::cout << frames1[i] << std::endl;
      }
      for (int i = 0; i < frames2.size(); i++) {
        std::cout << frames2[i] << std::endl;
      }

      std::cout << commonFrameIdx_ << " " << newFrameIdx_ << std::endl;

      return true;
    }

    bool GraphMerge::findCommonFrame(const std::vector<int> &frames1, const std::vector<int> &frames2) {
      int num_frames1 = frames1.size();
      int num_frames2 = frames2.size();
      if (num_frames2 != 2) {
        return false;
      }
      bool commonFrameFind = false;
      for (int i = 0; i < num_frames2; i++) {
        for (int j = 0; j < num_frames1; j++) {
          if (frames1[j] == frames2[i]) {
            if (commonFrameFind) {
              return false;
            }
            else {
              commonFrameIdx_ = j;
              commonFrameFind = true;
              break;
            }
          }
          else {
            newFrameIdx_ = j;
          }
        }
      }
      return commonFrameFind = true;
    }

} // namespace sparse_batch_sfm
