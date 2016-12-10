#include "BundleAdjustment.hpp"

#include <Eigen/Dense>

namespace sparse_batch_sfm {

  // BundleAdjustment::BundleAdjustment(const GraphStruct& graph) {
  //   

  //   return;
  // }

  bool BundleAdjustment::run(GraphStruct& graph) {
    // init
    num_cameras_ = graph.frame_idx.size();
    num_points_ = graph.Str.cols();
    num_parameters_ = 6 * num_cameras_ + 3 * num_points_;
    num_observations_ = graph.feature_idx.nonZeros();
    // std::cout << "Param:" << std::endl << num_cameras_ << "\n"
    // << num_points_ << "\n" << num_parameters_ << "\n" << num_observations_ << std::endl;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];
    parameters_ = new double[num_parameters_];

    // Filling in observations
    // (i, j, v) -> (camera ind, point ind, fvec_ind+1)
    int obs_ind = 0;
    for (int k = 0; k < graph.feature_idx.outerSize(); ++k) {
      for (Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(
                  graph.feature_idx, k); it; ++it) {
        int fvec_ind = it.value()-1;  // all + 1 in sparse matrix
        point_index_[obs_ind] = it.col();
        camera_index_[obs_ind] = it.row();
        // std::cout << "Sparse Matrix:" << std::endl;
        // std::cout << it.col() << ' ' << it.row() << std::endl; 
        // std::cout << fvec_ind << std::endl; 

        observations_[2 * obs_ind] =
            graph.feature_points[fvec_ind].pos(0);
        observations_[2 * obs_ind + 1] = 
            graph.feature_points[fvec_ind].pos(1);
        // std::cout << "observatios: " << std::endl;
        // std::cout << observations_[2 * obs_ind] << std::endl;
        // std::cout << observations_[2 * obs_ind + 1] << std::endl;
        ++ obs_ind;
      }
    }

    // Filling in parameters
    for (int i = 0; i < num_cameras_; ++i) {
      // Convert from rotation matrix to Angle-Axis representation
      // Requirement: graph.Mot[i] is column major
      double rotation[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
      Eigen::Map<Eigen::Matrix3d>(rotation, 3, 3) = graph.Mot[i].leftCols(3);
      // rotation[0] = graph.Mot[i](0, 0);
      // rotation[1] = graph.Mot[i](1, 0);
      // rotation[2] = graph.Mot[i](2, 0);
      // rotation[3] = graph.Mot[i](0, 1);
      // rotation[4] = graph.Mot[i](1, 1);
      // rotation[5] = graph.Mot[i](2, 1);
      // rotation[6] = graph.Mot[i](0, 2);
      // rotation[7] = graph.Mot[i](1, 2);
      // rotation[8] = graph.Mot[i](2, 2);
      double angleAxis[3] = {0, 0, 0};
      ceres::RotationMatrixToAngleAxis<double>(rotation, angleAxis);

      // std::cout << "Parameters: " << std::endl;
      // std::cout << angleAxis[0] << std::endl;
      // std::cout << angleAxis[1] << std::endl;
      // std::cout << angleAxis[2] << std::endl;
      // std::cout << graph.Mot[i](0, 3) << std::endl;
      // std::cout << graph.Mot[i](1, 3) << std::endl;
      // std::cout << graph.Mot[i](2, 3) << std::endl;
      
      parameters_[6 * i + 0] = angleAxis[0];
      parameters_[6 * i + 1] = angleAxis[1];
      parameters_[6 * i + 2] = angleAxis[2];
      parameters_[6 * i + 3] = graph.Mot[i](0, 3);
      parameters_[6 * i + 4] = graph.Mot[i](1, 3);
      parameters_[6 * i + 5] = graph.Mot[i](2, 3);

      // std::cout << parameters_[6 * i + 3] << parameters_[6 * i + 3] << 
    }

    for (int i = 0; i < num_points_; ++i) {
      parameters_[6 * num_cameras_ + i * 3]     = graph.Str(0, i);
      parameters_[6 * num_cameras_ + i * 3 + 1] = graph.Str(1, i);
      parameters_[6 * num_cameras_ + i * 3 + 2] = graph.Str(2, i);
    }

    // Optimize motion and structure
    ceres::Problem problem;
    const double* observations = this->observations();
    for (int i = 0; i < num_observations_; ++i) {
      // double K_arr[9];
      // Eigen::Map<Eigen::Matrix3d>(K_arr, 3, 3) = graph.K[camera_index_[i]].transpose(); // K_arr needs to be row major
      // K_arr[0] = graph.K[camera_index_[i]](0, 0);
      // K_arr[1] = graph.K[camera_index_[i]](0, 1);
      // K_arr[2] = graph.K[camera_index_[i]](0, 2);
      // K_arr[3] = graph.K[camera_index_[i]](1, 0);
      // K_arr[4] = graph.K[camera_index_[i]](1, 1);
      // K_arr[5] = graph.K[camera_index_[i]](1, 2);
      // K_arr[6] = graph.K[camera_index_[i]](2, 0);
      // K_arr[7] = graph.K[camera_index_[i]](2, 1);
      // K_arr[8] = graph.K[camera_index_[i]](2, 2);
      // std::cout << "K: ";
      // for (int i = 0; i < 9; ++i) {
      //   std::cout << K_arr[i] << ' ';
      // }
      // std::cout << std::endl;
      ceres::CostFunction* cost_function =
                SnavelyReprojectionError::Create(observations_[2*i],
                                                 observations_[2*i+1],
                                                 graph.K[camera_index_[i]]);
      problem.AddResidualBlock(cost_function,
                               NULL,
                               mutable_camera_for_observation(i),
                               mutable_point_for_observation(i));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    // Push back to graph
    for (int i = 0; i < num_cameras_; ++i) {
      // Convert from rotation matrix to Angle-Axis representation
      double* rotation = new double[9];
      ceres::AngleAxisToRotationMatrix<double>(parameters_ + 6 * i, rotation);
      graph.Mot[i].leftCols(3) = Eigen::Map<Eigen::MatrixXd>(rotation, 3, 3);
      // Eigen::MatrixXd tmp_matrix  = Eigen::Map<Eigen::MatrixXd>(rotation, 3, 3);
      // std::cout << tmp_matrix << std::endl;
      // graph.Mot[i].block(0, 0, 3, 3) = tmp_matrix;
      // graph.Mot[i](0, 0) = rotation[0];  
      // graph.Mot[i](1, 0) = rotation[1]; 
      // graph.Mot[i](2, 0) = rotation[2]; 
      // graph.Mot[i](0, 1) = rotation[3]; 
      // graph.Mot[i](1, 1) = rotation[4]; 
      // graph.Mot[i](2, 1) = rotation[5]; 
      // graph.Mot[i](0, 2) = rotation[6]; 
      // graph.Mot[i](1, 2) = rotation[7]; 
      // graph.Mot[i](2, 2) = rotation[8]; 
      // graph.Mot[i] << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

      // graph.Mot[i](0, 0) =0;
      // graph.Mot[i](1, 0) =0;  
      // graph.Mot[i](2, 0) =0;  
      // graph.Mot[i](0, 1) =0;  
      // graph.Mot[i](1, 1) =0;  
      // graph.Mot[i](2, 1) =0;  
      // graph.Mot[i](0, 2) =0;  
      // graph.Mot[i](1, 2) =0;  
      // graph.Mot[i](2, 2) =0;  

      graph.Mot[i](0, 3) = parameters_[6 * i + 3];
      graph.Mot[i](1, 3) = parameters_[6 * i + 4];
      graph.Mot[i](2, 3) = parameters_[6 * i + 5];
      delete[] rotation;
    }

    for (int i = 0; i < num_points_; ++i) {
      graph.Str(0, i) = parameters_[6 * num_cameras_ + i * 3];
      graph.Str(1, i) = parameters_[6 * num_cameras_ + i * 3 + 1];
      graph.Str(2, i) = parameters_[6 * num_cameras_ + i * 3 + 2];
    } 
    return true;
  }
}
