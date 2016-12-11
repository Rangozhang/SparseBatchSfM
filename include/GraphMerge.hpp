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
    bool inverseMot(const Eigen::MatrixXd& Mot, Eigen::MatrixXd& reveredMot);
    bool concatenateMots(const Eigen::MatrixXd& MotOuter, const Eigen::MatrixXd& MotInner, Eigen::MatrixXd& Mot);
    bool transformPtsByMot(const Eigen::MatrixXd& Mot, Eigen::Matrix<double, 6, Eigen::Dynamic>& Str);
    bool findCommonFrame(const std::vector<int> &frames1, const std::vector<int> &frames2);
  public: 
    bool merge(GraphStruct &graphA, const GraphStruct &graphB);
  private: 
    int commonFrameIdx1_;
    int commonFrameIdx2_;
    int newFrameIdx_;
};

}

#endif
