#ifndef GRAPHMERGE_HPP_
#define GRAPHMERGE_HPP_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "protos.hpp"

namespace sparse_batch_sfm {

class GraphMerge {
  private:
    Eigen::MatrixXd inverseMot(const Eigen::MatrixXd& Mot);
    Eigen::MatrixXd concatenateMots(const Eigen::MatrixXd& MotOuter, const Eigen::MatrixXd& MotInner);
    bool transformPtsByMot(const Eigen::MatrixXd& Mot, Eigen::Matrix<double, 6, Eigen::Dynamic>& Str);
    bool findCommonFrame(const std::vector<int> &frames1, const std::vector<int> &frames2);
  public: 
    bool merge(GraphStruct &graphA, GraphStruct &graphB);
  private: 
    int commonFrameIdx1_;
    int commonFrameIdx2_;
    int newFrameIdx_;
};

}

#endif
