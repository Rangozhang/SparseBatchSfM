#include "GraphMerge.hpp"
#include <unordered_map>
#include "opencv2/core/core.hpp"
#include "BundleAdjustment.hpp"

#include <Eigen/SVD>

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

  Eigen::MatrixXd GraphMerge::inverseMot(const Eigen::MatrixXd& Mot) {
    Eigen::MatrixXd inversedMot(3, 4);
    Eigen::Matrix3d R = Mot.leftCols(3);
    Eigen::MatrixXd t = Mot.rightCols(1);
    // inversedMot = [R', -R'*t]
    inversedMot.leftCols(3) = R.transpose();
    inversedMot.rightCols(1) = -R.transpose() * t;
    return inversedMot;
  }

  Eigen::MatrixXd GraphMerge::concatenateMots(const Eigen::MatrixXd& MotOuter, const Eigen::MatrixXd& MotInner) {
    Eigen::MatrixXd Mot(3, 4);
    // Mot * X = MotOuter * MotInner * X
    Eigen::Matrix3d Ro = MotOuter.leftCols(3);
    Eigen::Matrix3d Ri = MotInner.leftCols(3);
    Eigen::MatrixXd to = MotOuter.rightCols(1);
    Eigen::MatrixXd ti = MotInner.rightCols(1);
    Mot.leftCols(3) = Ro * Ri;
    Mot.rightCols(1) = Ro * ti + to;
    return Mot;
  }

  bool GraphMerge::transformPtsByMot(const Eigen::MatrixXd& Mot, Eigen::Matrix<double, 6, Eigen::Dynamic>& Str) {
    if (Mot.rows() != 3 || Mot.cols() != 4) {
      std::cout << "Invalid Mot" << std::endl;
      return false;
    }
    int str_len = Str.cols();
    Str.topRows(3) = Mot.leftCols(3) * Str.topRows(3) + Mot.rightCols(1).replicate(1, str_len);
    return true;
  }

  bool GraphMerge::multiTriangulate(GraphStruct& graph) {
    int n_frames = graph.frame_idx.size();
    int n_3dpt = graph.feature_idx.cols();
    // For each Point, use SVD to recover the 3d P
    for (int i = 0; i < n_3dpt; ++i) {
      std::vector<int> validFrameIdx = {};
      std::vector<int> featureIdxForValidFrame = {};
      for (int j = 0; j < n_frames; ++j) {
        int feature_point_idxp1 = graph.feature_idx.coeff(j, i);
        if (feature_point_idxp1 > 0) {
          validFrameIdx.push_back(j);
          featureIdxForValidFrame.push_back(feature_point_idxp1-1);
        }
      }
      int n_valid_frames = featureIdxForValidFrame.size();
      Eigen::MatrixXd A(2*n_valid_frames, 4);

      for (int k = 0; k < n_valid_frames; ++k) {
        // get the pos
        int x = graph.feature_points[featureIdxForValidFrame[k]].pos(0);
        int y = graph.feature_points[featureIdxForValidFrame[k]].pos(1);

        Eigen::MatrixXd Mi = graph.K[validFrameIdx[k]] * graph.Mot[validFrameIdx[k]];

        // [x*m_3' - m_1']
        // [y*m_3' - m_2'] P = 0
        // [x*m_3' - m_1']
        // [y*m_3' - m_2']
        A.row(2*k)   = x * Mi.row(2) - Mi.row(0);
        A.row(2*k+1) = y * Mi.row(2) - Mi.row(1);
      }

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::MatrixXd raw_str = svd.matrixV().rightCols(1);
      raw_str = raw_str / raw_str(3, 0);
      graph.Str.block(0, i, 3, 1) = raw_str.topRows(3);
    }
    return true;
  }

  bool GraphMerge::merge(GraphStruct &graphA, GraphStruct &graphB) {
    // find common frame and new frame
    if (!findCommonFrame(graphA.frame_idx, graphB.frame_idx)) {
      return false;
    }
    int num_frameA = graphA.frame_idx.size();

    std::cout << "graph A frames: ";
    for (int i = 0; i < graphA.frame_idx.size(); i++) {
      std::cout << graphA.frame_idx[i] << " ";
    }
    std::cout << std::endl << "graph B frames: ";
    for (int i = 0; i < graphB.frame_idx.size(); i++) {
      std::cout << graphB.frame_idx[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "common frame index 1: " << commonFrameIdx1_ << "; common frame index 2: " << commonFrameIdx2_ << "; new frame index 2: " << newFrameIdx_ << std::endl;

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
    std::vector<Eigen::Triplet<int>> triplet;
    std::vector<int> unseen_idx;
    for (int i = 0; i < graphA.feature_idx.outerSize(); i++) {
      for (Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(graphA.feature_idx, i); it; ++it) {
        triplet.push_back(Eigen::Triplet<int>(it.row(), it.col(), it.value()));
      }
    }
    /*
    for (int i = 0; i < triplet.size(); i++) {
      std::cout << triplet[i].row() << " " << triplet[i].col() << " " << triplet[i].value() << std::endl;
    }
    */
    int num_point2 = graphB.feature_idx.cols();
    int count = 0;
    std::vector<int> str1, str2;
    for (int i = 0; i < num_point2; i++) {
      int point_ind2 = graphB.feature_idx.coeff(commonFrameIdx2_, i);
      if (point_ind2 > 0) {
        point_2d this_point;
        this_point.x = graphB.feature_points[point_ind2 - 1].pos(0);
        this_point.y = graphB.feature_points[point_ind2 - 1].pos(1);
        int point_ind1 = graphB.feature_idx.coeff(newFrameIdx_, i);
        if (hash_point.count(this_point)) {
          graphA.feature_points.push_back(graphB.feature_points[point_ind1 - 1]);
          triplet.push_back(Eigen::Triplet<int>(num_frameA, hash_point[this_point], graphA.feature_points.size()));

          // Eigen::MatrixXd dist1 = M1.leftCols(3) * (graphA.Str.block(0,hash_point[this_point],3,1)-M1.rightCols(1));
          // str1.push_back(graphA.Str.block(0,hash_point[this_point],3,1));
          str1.push_back(hash_point[this_point]);

          // Eigen::MatrixXd dist2 = M2.leftCols(3) * (graphB.Str.block(0,i,3,1)-M2.rightCols(1));
          // str2.push_back(graphB.Str.block(0,i,3,1));
          str2.push_back(i);

          count++;
        }
        else {
          graphA.feature_points.push_back(graphB.feature_points[point_ind2 - 1]);
          triplet.push_back(Eigen::Triplet<int>(commonFrameIdx1_, num_point1, graphA.feature_points.size()));
          graphA.feature_points.push_back(graphB.feature_points[point_ind1 - 1]);
          triplet.push_back(Eigen::Triplet<int>(num_frameA, num_point1, graphA.feature_points.size()));
          unseen_idx.push_back(i);
          num_point1++;
        }
      }
    }

    double averaged_ratio = 0;
    for (int i = 0; i < str1.size(); ++i) {
      Eigen::MatrixXd M1 = graphA.Mot[commonFrameIdx1_];
      Eigen::MatrixXd dis1 = M1.leftCols(3) * (graphA.Str.block(0, str1[i], 3, 1) - M1.rightCols(1));
      Eigen::MatrixXd M2 = graphB.Mot[commonFrameIdx2_];
      Eigen::MatrixXd dis2 = M2.leftCols(3) * (graphB.Str.block(0, str2[i], 3, 1) - M2.rightCols(1));
      // std::cout << "GraphA/GraphB dist:";
      // std::cout << dis1.norm() << ' '<< dis2.norm()
      //   << " ptA: " << graphA.Str(0, str1[i]) << ' ' << graphA.Str(1, str1[i]) << ' ' << graphA.Str(2, str1[i])
      //   << " ptB: " << graphB.Str(0, str2[i]) << ' ' << graphB.Str(1, str2[i]) << ' ' << graphB.Str(2, str2[i])
      //   << std::endl;
      averaged_ratio += dis1.norm()/dis2.norm();
    }
    averaged_ratio /= str1.size();

    std::cout << "Avergaed scale ratio: " << averaged_ratio << std::endl;
    for (int i = 0; i < graphB.Mot.size(); ++i) {
      graphB.Mot[i].rightCols(1) = graphB.Mot[i].rightCols(1) * averaged_ratio;
    }

    // re-triangulation
    multiTriangulate(graphB);
    // BundleAdjustment ba;
    // ba.run(graphB);

    
    /*
    for (int i = 0; i < triplet.size(); i++) {
      std::cout << triplet[i].row() << " " << triplet[i].col() << " " << triplet[i].value() << std::endl;
    }
    for (int i = 0; i < unseen_idx.size(); i++) {
      std::cout << unseen_idx[i] << std::endl;
    }
    */
    std::cout << "Number of common points " << count << "/" << num_point2 << std::endl;

    // transform graphB into the world coordinate of graphA
    Eigen::MatrixXd MotBwAw = concatenateMots(inverseMot(graphA.Mot[commonFrameIdx1_]), graphB.Mot[commonFrameIdx2_]);
    // std::cout << "Motion for graphB into the world coordinate of graphA: " << std::endl;
    // std::cout << MotBwAw << std::endl;

    // std::cout << "GraphA Structure:" << std::endl;
    // std::cout << graphA.Str.leftCols(5) << std::endl;
    // std::cout << "GraphB Structure:" << std::endl;
    // std::cout << graphB.Str.leftCols(5) << std::endl;
    if (!transformPtsByMot(MotBwAw, graphB.Str)) {
      return false;
    }
    // std::cout << "GraphB Structure after transformation:" << std::endl;
    // std::cout << graphB.Str.leftCols(5) << std::endl;

    for (int i = 0; i < str1.size(); ++i) {
      Eigen::MatrixXd M1 = graphA.Mot[commonFrameIdx1_];
      Eigen::MatrixXd dis1 = M1.leftCols(3) * (graphA.Str.block(0, str1[i], 3, 1) - M1.rightCols(1));
      Eigen::MatrixXd M2 = graphB.Mot[commonFrameIdx2_];
      Eigen::MatrixXd dis2 = M2.leftCols(3) * (graphB.Str.block(0, str2[i], 3, 1) - M2.rightCols(1));
      // std::cout << "GraphA/GraphB dist (re-scaled):";
      // std::cout << dis1.norm() << ' '<< dis2.norm()
      //   << " ptA: " << graphA.Str(0, str1[i]) << ' ' << graphA.Str(1, str1[i]) << ' ' << graphA.Str(2, str1[i])
      //   << " ptB: " << graphB.Str(0, str2[i]) << ' ' << graphB.Str(1, str2[i]) << ' ' << graphB.Str(2, str2[i])
      //   << std::endl;
    }

    graphA.frame_idx.push_back(graphB.frame_idx[newFrameIdx_]);
    graphA.Mot.push_back(concatenateMots(graphB.Mot[newFrameIdx_], inverseMot(MotBwAw)));
    graphA.K.push_back(graphB.K[newFrameIdx_]);
    graphA.feature_idx.resize(num_frameA + 1, num_point1);
    graphA.feature_idx.setFromTriplets(triplet.begin(), triplet.end());
    Eigen::Matrix<double, 6, Eigen::Dynamic> Str_temp = graphA.Str;
    graphA.Str.resize(6, num_point1);
    graphA.Str.leftCols(Str_temp.cols()) = Str_temp;
    for (int i = 0; i < unseen_idx.size(); i++) {
      graphA.Str.col(Str_temp.cols() + i) = graphB.Str.col(unseen_idx[i]);
    }

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
