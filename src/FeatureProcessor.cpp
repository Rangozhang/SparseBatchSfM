#include <unordered_set>

#include "FeatureProcessor.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"

namespace sparse_batch_sfm {
  bool FeatureProcessor::feature_match(const std::vector<std::unique_ptr<cv::Mat>>& image_seq, 
                                       FeatureStruct& feature_struct, 
                                       int minHessian = 400, float match_thres = 200, bool visualize = false) {

  // feature_struct initializtion
  int seq_len = image_seq.size();
  feature_struct.skeleton.resize(seq_len, seq_len);
  for (int i = 0; i < seq_len; i++) {
    feature_struct.skeleton(i, i) = 0;
  }
  
  // SIFT detector and extractor
  cv::SiftFeatureDetector detector(minHessian);
  cv::SiftDescriptorExtractor extractor;
  
  // detect keypoints and extract features for all images
  std::vector<std::vector<cv::KeyPoint>> keypoints;
  std::vector<cv::Mat> descriptors;
  for (int i = 0; i < seq_len; i++) {
    std::vector<cv::KeyPoint> this_keypoint;
    detector.detect(*image_seq[i].get(), this_keypoint);
    cv::Mat this_descriptor;
    extractor.compute(*image_seq[i].get(), this_keypoint, this_descriptor);
    keypoints.push_back(this_keypoint);
    descriptors.push_back(this_descriptor);
  }

  // match features between each pair of images
  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;
  for (int i = 0; i < seq_len; i++) {
    for (int j = i + 1; j < seq_len; j++) {
      matcher.match(descriptors[i], descriptors[j], matches);
      // count the number of good matches
      std::vector<cv::DMatch> good_matches;
      int count = 0;
      for(int n = 1; n < matches.size(); n++) {
        if(matches[n].distance < match_thres) {
          if (visualize) {
            good_matches.push_back(matches[n]);
          }
          count++;
        }
      }
      feature_struct.skeleton(i,j) = count;
      feature_struct.skeleton(j,i) = count;
      //std::cout << matches[n].imgIdx << ' ' << matches[n].trainIdx << ' ' << matches[n].queryIdx << ' ' << matches[n].distance << std::endl;
      // show matches
      if (visualize) {
        cv::Mat img_matches;
        cv::drawMatches(*image_seq[i].get(), keypoints[i], *image_seq[j].get(), keypoints[j],
                        good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cv::imshow("Matches", img_matches);
        cv::waitKey(0);
      }
    }
  }
  
  return true;
  }

  inline void print_unordered_set(const std::unordered_set<int>& mset) {
    for (const auto& ele : mset) {
        std::cout << " " << ele;
    }
    std::cout << std::endl;
    return;
  }

  bool FeatureProcessor::skeletonize(Eigen::Matrix<int, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>& skeleton, float min_n_matches=20) {
    int len = skeleton.rows();
    bool verbose = false;
    Eigen::MatrixXi mask(len, len);
    mask.setZero();
    auto bin_mat_ = (skeleton.array() > min_n_matches); // readonly... XD
    Eigen::MatrixXi bin_mat(len, len);
    for (int i = 0; i < len; ++i) {
      for (int j = 0; j < len; ++j) {
        bin_mat(i, j) = bin_mat_(i, j);
      }
    }
    if (verbose) {
      std::cout << "binary adjacent matrix: " << std::endl
                << bin_mat << std::endl << std::endl;
    }
    std::unordered_set<int> picked;
    std::unordered_set<int> candidate;

    int init_ind = -1, pre_ind = -1;
    double init_val = bin_mat.rowwise().sum().col(0).maxCoeff(&init_ind);
    if (verbose) {
      std::cout << "init_val: " << init_val << " init_ind: " << init_ind << std::endl;
    }
    if (init_ind < 0) {
      return false;
    }
    picked.insert(init_ind);
    pre_ind = init_ind;
    bin_mat.col(pre_ind).setZero();
    int cur_ind = -1;

    // CDS
    do {
      // 1. Push neighbors in to candidate
      if (verbose) {
        std::cout << "Status: " << std::endl;
        std::cout << "bin_mat: " << std::endl << bin_mat << std::endl;
        std::cout << "pre_ind: " << pre_ind << std::endl;
        std::cout << "cur_ind: " << cur_ind << std::endl;
        std::cout << "picked: "; print_unordered_set(picked);
        std::cout << "candidate: "; print_unordered_set(candidate);
        std::cout << "mask: " << std::endl << mask << std::endl;
        std::cout << std::endl;
      }
      for (int i = 0; i < len; ++i) {
        // continue if diagonal / value=0 / in picked set
        if (i == pre_ind || !bin_mat(pre_ind, i)) {
          continue;
        }
        candidate.insert(i);
        bin_mat.col(i).setZero();
      }
      
      // 2. push cur_ind to picked from candidate
      cur_ind = -1;
      double cur_val = 0;
      for (const int& ele : candidate) {
        int n_lin = bin_mat.rowwise().sum()(ele);
        if (cur_val < n_lin) {
          cur_ind = ele;
          cur_val = n_lin;
        }
      }
      if (cur_ind == -1) {
          break;
      }
      candidate.erase(cur_ind);
      picked.insert(cur_ind);
      
      // 3. store edge in mask
      mask(pre_ind, cur_ind) = 1;
      mask(cur_ind, pre_ind) = 1;

      // 4. set cur_ind to pre_ind
      pre_ind = cur_ind;
      if (verbose) {
        std::cout << "Status: " << std::endl;
        std::cout << "bin_mat: " << std::endl << bin_mat << std::endl;
        std::cout << "pre_ind: " << pre_ind;
        std::cout << "cur_ind: " << cur_ind;
        std::cout << "picked: "; print_unordered_set(picked);
        std::cout << "candidate: "; print_unordered_set(candidate);
        std::cout << "mask: " << std::endl << mask << std::endl;
        std::cout << std::endl;
      }
 
    } while (!candidate.empty());

    // adding leaves
    for (int i = 0; i < len; ++i) {
      if (picked.count(i)) continue;
      int cds_ind = -1;
      int con_weight = 0;
      for (const int& ele : picked) {
        int cur_w = skeleton(ele, i);
        if (cur_w > con_weight) {
          cds_ind = ele;
          con_weight = cur_w;
        }
      }
      if (cds_ind == -1) continue;
      mask(i, cds_ind) = 1;
      mask(cds_ind, i) = 1;
    }

    skeleton = skeleton.cwiseProduct(mask);

    if (verbose) {
      std::cout << "Status: " << std::endl;
      std::cout << "bin_mat: " << std::endl << bin_mat << std::endl;
      std::cout << "pre_ind: " << pre_ind;
      std::cout << "cur_ind: " << cur_ind;
      std::cout << "picked: "; print_unordered_set(picked);
      std::cout << "candidate: "; print_unordered_set(candidate);
      std::cout << "mask: " << std::endl << mask << std::endl;
      std::cout << std::endl;
    }
 
    return true;
  }
  
}
