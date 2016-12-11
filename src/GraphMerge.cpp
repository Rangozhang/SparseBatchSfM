#include "GraphMerge.hpp"

namespace sparse_batch_sfm {
    bool inverseMot(const Eigen::MatrixXd& Mot, Eigen::MatrixXd& reveredMot) {
      if (Mot.rows() != 3 || Mot.cols() != 4) {
        std::cout << "Invalid Mot" << std::endl;
        return false;
      }
      reveredMot.resize(3, 4);
      Eigen::Matrix3d R = Mot.leftCols(3);
      Eigen::MatrixXd t = Mot.rightCols(1);
      // reveredMot = [R', -R'*t]
      reveredMot.leftCols(3) = R.transpose();
      reveredMot.rightCols(1) = -R.transpose() * t;
      return true;
    }

    bool concatenateMots(const Eigen::MatrixXd& MotOuter,
                        const Eigen::MatrixXd& MotInner, Eigen::MatrixXd& Mot) {
      if (MotOuter.rows() != 3 || MotOuter.cols() != 4 || MotInner.rows() != 3 || MotInner.cols() != 4) {
        std::cout << "Invalid Mot" << std::endl;
        return false;
      }
      Mot.resize(3, 4);
      // Mot * X = MotOuter * MotInner * X
      Eigen::Matrix3d Ro = MotOuter.leftCols(3);
      Eigen::Matrix3d Ri = MotOuter.leftCols(3);
      Eigen::MatrixXd to = MotInner.rightCols(1);
      Eigen::MatrixXd ti = MotInner.rightCols(1);
      Mot.leftCols(3) = Ro * Ri;
      Mot.rightCols(1) = Ro * ti + to;
      return true;
    }

    bool transfromPtsByMot(const Eigen::MatrixXd& Mot, Eigen::Matrix<double, 6, Eigen::Dynamic>& Str) {
      if (Mot.rows() != 3 || Mot.cols() != 4) {
        std::cout << "Invalid Mot" << std::endl;
        return false;
      }
      int str_len = Str.cols();
      Str.topRows(3) = Mot.leftCols(3) * Str.topRows(3) + Mot.rightCols(1).replicate(1, str_len);
      return true;
    }

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
