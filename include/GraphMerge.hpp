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
    // void concatenateRts(RtOuter, RtInner, )
};

}

#endif
