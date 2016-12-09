#ifndef BUNDLEADJUSTMENT_HPP_
#define BUNDLEADJUSTMENT_HPP_

/* reference: https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/simple_bundle_adjuster.cc
 * 
 */

#include <iostream>
#include <memory>

#include "protos.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"


namespace sparse_batch_sfm {

namespace {

constexpr int N_PAR_CAM = 6;

}

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 6 parameters: 3 for rotation, 3 for translation
struct SnavelyReprojectionError {
  // The parameter intrinsic is a 9-by-1 vector for K (Row Majored)
  SnavelyReprojectionError(double observed_x, double observed_y, double* K)
      : observed_x(observed_x), observed_y(observed_y), K(K) {}
  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T predicted_x = T(K[0]) * p[0] + T(K[1]) * p[1] + T(K[2]) * p[2];
    T predicted_y = T(K[3]) * p[0] + T(K[4]) * p[1] + T(K[5]) * p[2];
    T predicted_z = T(K[6]) * p[0] + T(K[7]) * p[1] + T(K[8]) * p[2];

    // Normalization
    predicted_x /= predicted_z;
    predicted_y /= predicted_z;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
									 double* K) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, N_PAR_CAM, 3>(
                new SnavelyReprojectionError(observed_x, observed_y, K)));
  }
  double observed_x;
  double observed_y;
  double* K;
};

class BundleAdjustment {
 public:

  // BundleAdjustment();
  ~BundleAdjustment() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }

  bool run(GraphStruct& graph);
  int num_observations()       const { return num_observations_;               }
  const double* observations() const { return observations_;                   }
  double* mutable_cameras()          { return parameters_;                     }
  double* mutable_points()           { return parameters_  + N_PAR_CAM * num_cameras_; }
  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * N_PAR_CAM;
  }
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  } 

 private:

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  double* parameters_;   // N_PAR_CAM * camera + points
};

} // namespace sparse_batch_sfm

#endif
