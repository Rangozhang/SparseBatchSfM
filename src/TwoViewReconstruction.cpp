#include "TwoViewReconstruction.hpp"

#include <unordered_set>

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <Eigen/SVD>

namespace sparse_batch_sfm {

namespace {
  bool DEBUG = true;
}

  bool TwoViewReconstruction::reconstruct(FeatureStruct& feature_struct, int frame1, int frame2,
                                          int width, int height,
                                          Eigen::Matrix3d K1, Eigen::Matrix3d K2, GraphStruct& graph,
										  const cv::Mat& img1, const cv::Mat& img2) {

    // 1. compute F
    if (!estimateF(feature_struct, frame1, frame2, width, height, img1, img2)) {
      std::cout << "Failed on ransacF" << std::endl;
      return false;
    }

    // 2. compute E
    E_ = graph.K[frame1].transpose() * F_ * graph.K[frame2];

    // 3. Mot from E
    graph.Mot.push_back(RtFromE(graph.K[frame1], graph.K[frame2], feature_struct, frame1, frame2));

    // 4. Triangulation
    if (!triangulate(graph)) {
      std::cout << "Failed on triangulation" << std::endl;
      return false;
    }

    return true;
  }

  bool TwoViewReconstruction::estimateF(FeatureStruct& feature_struct,
                                        int frame1, int frame2, int img_width, int img_height, const cv::Mat& img1, const cv::Mat& img2) {
    std::vector<cv::Point2f> pts1, pts2;
    for (int i = 0; i < feature_struct.feature_matches[frame1][frame2].size(); ++i) {
      cv::Point2f tmp_pt;
      int frame1_pt_ind = feature_struct.feature_matches[frame1][frame2][i].row();
      tmp_pt.x = feature_struct.feature_point[frame1][frame1_pt_ind].pos(0);
      tmp_pt.y = feature_struct.feature_point[frame1][frame1_pt_ind].pos(1);
      tmp_pt.x /= img_width + 0.0;
      tmp_pt.y /= img_height + 0.0;
      pts1.push_back(tmp_pt);

      int frame2_pt_ind = feature_struct.feature_matches[frame1][frame2][i].col();
      tmp_pt.x = feature_struct.feature_point[frame2][frame2_pt_ind].pos(0);
      tmp_pt.y = feature_struct.feature_point[frame2][frame2_pt_ind].pos(1);
      tmp_pt.x /= img_width + 0.0;
      tmp_pt.y /= img_height + 0.0;
      pts2.push_back(tmp_pt);
    }

    cv::Mat scale = cv::Mat::zeros(3, 3, CV_64FC1);
    scale.at<double>(0, 0) = 1.0 / img_width;
    scale.at<double>(1, 1) = 1.0 / img_height;
    scale.at<double>(2, 2) = 1.0;

    cv::Mat mask;
    cv::Mat f_mat = cv::findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 0.001, 0.99, mask);
    f_mat = scale * f_mat * scale;

    cv::cv2eigen(f_mat, F_);
/*    for (int i = 0; i < feature_struct.feature_matches[frame1][frame2].size(); ++i) {
      cv::Point2f tmp_pt1, tmp_pt2;
      int frame1_pt_ind = feature_struct.feature_matches[frame1][frame2][i].row();
      tmp_pt1.x = feature_struct.feature_point[frame1][frame1_pt_ind].pos(0);
      tmp_pt1.y = feature_struct.feature_point[frame1][frame1_pt_ind].pos(1);

      int frame2_pt_ind = feature_struct.feature_matches[frame1][frame2][i].col();
      tmp_pt2.x = feature_struct.feature_point[frame2][frame2_pt_ind].pos(0);
      tmp_pt2.y = feature_struct.feature_point[frame2][frame2_pt_ind].pos(1);
      std::cout << Eigen::Vector3d(tmp_pt1.x, tmp_pt1.y, 1).transpose() * f_mat * Eigen::Vector3d(tmp_pt2.x, tmp_pt2.y, 1) << std::endl;
    }*/

    int count = 0;
    // Get rid of outliers from ransacF
    for (int i = mask.rows-1; i >= 0; --i) {
      // if it's outlier
      if (mask.at<char>(i, 0) == 0) {
        feature_struct.feature_matches[frame1][frame2].erase(
                feature_struct.feature_matches[frame1][frame2].begin() + i);
      }
      else
        count ++;
    }
    std::cout << mask.size() << ' ' << count << std::endl;

    if (DEBUG) {
   	  /* draw epolir line */
      using namespace cv;
	  cv::Mat Epilines;
	  cv::computeCorrespondEpilines(Mat(pts1),1,f_mat,Epilines);
	  // cout<<"size of Epilines: "<<Epilines.rows<<" | "<<Epilines.cols<<endl;
	  // cout<<"depth of Epilines: "<<Epilines.depth()<<endl;

	  Point3f top_horizontal = Point3f(0,1,0);
	  Point3f left_vertical  =   Point3f(1, 0, 0);
	  Point3f bottom_horizontal = Point3f(0, 1, -img_height);
	  Point3f right_vertical =    Point3f(1, 0, -img_width);

	  Mat Epilines_show;
	  img2.copyTo(Epilines_show);
	  for(int i = 0; i < Epilines.rows; i++){
	  	Point2f A;
	  	Point2f B;

	  	Point3f Eline = Point3f(Epilines.at<float>(i,0),Epilines.at<float>(i,1),Epilines.at<float>(i,2));
	  	Point3f candidate_1 = top_horizontal.cross(Eline);
	  	Point2f candidate_1_cord = Point2f(candidate_1.x/candidate_1.z, candidate_1.y/candidate_1.z);
	  	if(candidate_1_cord.x >= 0 && candidate_1_cord.x <= img_width) {
	  		/* code */
	  		A = candidate_1_cord;
	  	}
	  	Point3f candidate_2 = left_vertical.cross(Eline);
	  	Point2f candidate_2_cord = Point2f(candidate_2.x/candidate_2.z, candidate_2.y/candidate_2.z);
	  	if(candidate_2_cord.y >= 0 && candidate_2_cord.y <= img_width) {
	  		/* code */
	  		A = candidate_2_cord;
	  	}
	  	Point3f candidate_3 = bottom_horizontal.cross(Eline);
	  	Point2f candidate_3_cord = Point2f(candidate_3.x/candidate_3.z, candidate_3.y/candidate_3.z);
	  	if(candidate_3_cord.x >= 0 && candidate_3_cord.x <= img_width) {
	  		/* code */
	  		B = candidate_3_cord;
	  	}
	  	Point3f candidate_4 = right_vertical.cross(Eline);
	  	Point2f candidate_4_cord = Point2f(candidate_4.x/candidate_4.z, candidate_4.y/candidate_4.z);
	  	if(candidate_4_cord.y >= 0 && candidate_4_cord.y <= img_width) {
	  		/* code */
	  		B = candidate_4_cord;
	  	}

	  	line(Epilines_show,A,B,Scalar(0,255,0));
	  	// cout<<"point: "<<x<<"  "<<y<<endl;
	  }
	  Mat Epilines_show_pts;

	  std::vector<KeyPoint> pts2_kp;
	  KeyPoint::convert(pts2, pts2_kp);
	  drawKeypoints( Epilines_show, pts2_kp, Epilines_show_pts, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	  imshow("Epilines_show",Epilines_show_pts);
	  waitKey(0);   
    }

    return true;
  }

  Eigen::Matrix<double, 3, 4, Eigen::ColMajor>
                        TwoViewReconstruction::RtFromE(const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2,
                                                       const FeatureStruct& feature_struct,
                                                       int frame1, int frame2) {
    Eigen::Matrix<double, 3, 4, Eigen::ColMajor> Mot;

    // 1. Get Motion matrix candidates
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    Eigen::Matrix3d R1 = U * W * V.transpose();
    Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();
    Eigen::MatrixXd t1 = U.col(3);    // TODO: ask god leg if need to be normalized
    Eigen::MatrixXd t2 = -U.col(3);

    if (R1.determinant() < 0) {
      R1 = -R1;
    }
    if (R2.determinant() < 0) {
      R2 = -R2;
    }

    std::vector<Eigen::Matrix<double, 3, 4, Eigen::ColMajor>> mot_candidate(4);
    mot_candidate[0] << R1, t1;
    mot_candidate[1] << R1, t2;
    mot_candidate[2] << R2, t1;
    mot_candidate[3] << R2, t2;

    // 2. Find the best one with most inliers by triangulation

    return Mot;
  }

  bool TwoViewReconstruction::triangulate(GraphStruct& graph) {
    return true;
  }

}
