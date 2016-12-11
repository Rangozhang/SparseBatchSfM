#ifndef GRAPHMERGE_HPP_
#define GRAPHMERGE_HPP_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "proto.hpp"

namespace sparse_batch_sfm {

class GraphMerge {
  private:
    void inverseRt(const MatrixXd& Rt, MatrixXd& reveredRt);
    // void concatenateRts(RtOuter, RtInner, )
}

}

#endif
