#include "GraphMerge.hpp"
#include <unordered_map>
#include "opencv2/core/core.hpp"

struct point_2d {
  double x, y;
};
bool operator == (const point_2d & p1, const point_2d & p2) {
  return p1.x == p2.x && p1.y == p2.y;
}
namespace std {
  template<> struct hash<point_2d>
  {
    std::size_t operator()(point_2d const& p) const
    {
      std::size_t const h1 (std::hash<double>{}(p.x));
      std::size_t const h2 (std::hash<double>{}(p.y));
      return h1 ^ (h2 << 1);
    }
  };
}

namespace sparse_batch_sfm {

    bool GraphMerge::merge(GraphStruct &graphA, const GraphStruct &graphB) {
      // find common frame and new frame
      if (!findCommonFrame(graphA.frame_idx, graphB.frame_idx)) {
        return false;
      }

      // transform graphB into the world coordinate of graphA


      // update the merged graph
      // put the points in the common frame into hash table
      std::unordered_map<point_2d, int> hash_point;
      int num_point1 = graphA.feature_idx.cols();
      for (int i = 0; i < num_point1; i++) {
        int point_ind = graphA.feature_idx.coeff(commonFrameIdx1_, i);
        if (point_ind > 0) {
          point_2d this_point;
          this_point.x = graphA.feature_points[point_ind - 1].pos(0);
          this_point.y = graphA.feature_points[point_ind - 1].pos(1);
          hash_point.insert({this_point, i});
        }
      }

      // find whether a point in graphB is seen in graphA
      int num_point2 = graphB.feature_idx.cols();
      int count = 0;
      for (int i = 0; i < num_point2; i++) {
        int point_ind = graphB.feature_idx.coeff(commonFrameIdx2_, i);
        if (point_ind > 0) {
          point_2d this_point;
          this_point.x = graphB.feature_points[point_ind - 1].pos(0);
          this_point.y = graphB.feature_points[point_ind - 1].pos(1);
          std::unordered_map<point_2d, int>::const_iterator seen = hash_point.find(this_point);
          if (seen == hash_point.end()) {
            count++;
          }
          else {

          }
        }
      }
      std::cout << num_point2 - count << "/" << num_point2 << std::endl;

      for (int i = 0; i < graphA.frame_idx.size(); i++) {
        std::cout << graphA.frame_idx[i] << std::endl;
      }
      for (int i = 0; i < graphB.frame_idx.size(); i++) {
        std::cout << graphB.frame_idx[i] << std::endl;
      }

      std::cout << commonFrameIdx1_ << " " << commonFrameIdx2_ << " " << newFrameIdx_ << std::endl;

      return true;
    }

    bool GraphMerge::findCommonFrame(const std::vector<int> &frames1, const std::vector<int> &frames2) {
      int num_frames1 = frames1.size();
      int num_frames2 = frames2.size();
      if (num_frames2 != 2) {
        std::cout << "More than 2 frames founded in the new graph" << std::endl;
        return false;
      }
      bool commonFrameFind = false;
      for (int i = 0; i < num_frames2; i++) {
        bool isCommonFrame = false;
        for (int j = 0; j < num_frames1; j++) {
          if (frames1[j] == frames2[i]) {
            if (commonFrameFind) {
              std::cout << "Common frame founded twice in the new graph" << std::endl;
              return false;
            }
            else {
              commonFrameIdx1_ = j;
              commonFrameIdx2_ = i;
              commonFrameFind = true;
              isCommonFrame = true;
              break;
            }
          }
        }
        if (!isCommonFrame) {
          newFrameIdx_ = i;
        }
      }
      if (!commonFrameFind) {
        std::cout << "Common frame not founded in the new graph" << std::endl;
        return false;
      }
      return true;
    }

} // namespace sparse_batch_sfm
