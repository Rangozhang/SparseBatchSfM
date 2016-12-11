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
    void inverseRt(const Eigen::MatrixXd& Rt, Eigen::MatrixXd& reveredRt);
    void concatenateRts(const Eigen::MatrixXd& RtOuter, const Eigen::MatrixXd& RtInner, Eigen::MatrixXd& Rt);
    void transformPtsByRt(const Eigen::MatrixXd& X3D, const Eigen::MatrixXd& Rt, Eigen::MatrixXd& Y3D);
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
