#include <unordered_set>
#include <unordered_map>

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
    std::vector<FeaturePoint> features;
    for (int n = 0; n < this_keypoint.size(); n++) {
      FeaturePoint feature_point;
      cv::Point2f pos = keypoints[i][n].pt;
      feature_point.pos = Eigen::Vector2d(pos.x, pos.y);
      cv::Vec3i rgb = image_seq[i].get()->at<cv::Vec3b>(round(feature_point.pos[1]), round(feature_point.pos[0]));
      feature_point.rgb = Eigen::Vector3i(rgb[2], rgb[1], rgb[0]);
      features.push_back(feature_point);
    }
    feature_struct.feature_point.push_back(features);
  }

  // match features between each pair of images
  std::vector<std::unordered_map<int, int>> hash(seq_len);
  std::vector<Eigen::Triplet<int>> triplet;
  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;
  int count_point = 0;
  for (int i = 0; i < seq_len; i++) {
    std::vector<std::vector<Eigen::Triplet<double>>> this_match;
    for (int j = 0; j < seq_len; j++) {
      std::vector<Eigen::Triplet<double>> this_this_match;
      if (j > i) {
        matcher.match(descriptors[i], descriptors[j], matches);
        // count the number of good matches
        std::vector<cv::DMatch> good_matches;
        int count = 0;
        for(int n = 1; n < matches.size(); n++) {
          if(matches[n].distance < match_thres) {
            this_this_match.push_back(Eigen::Triplet<double>(matches[n].queryIdx, matches[n].trainIdx));
            good_matches.push_back(matches[n]);
            count++;
/*
          std::unordered_map<int, int>::const_iterator got_i = hash[i].find(matches[n].queryIdx);
          if (got_i == hash[i].end()) {
            FeaturePoint feature_point;
            cv::Point2f pos = keypoints[i][matches[n].queryIdx].pt;
            feature_point.pos = Eigen::Vector2d(pos.x, pos.y);
            cv::Vec3i rgb = image_seq[i].get()->at<cv::Vec3b>(round(feature_point.pos[1]), round(feature_point.pos[0]));
            feature_point.rgb = Eigen::Vector3i(rgb[2], rgb[1], rgb[0]);
            feature_struct.feature_point.push_back(feature_point);
            hash[i].insert({matches[n].queryIdx, feature_struct.feature_point.size()});
          }
          int idx_i = feature_struct.feature_point.size();
          std::unordered_map<int, int>::const_iterator got_j = hash[j].find(matches[n].trainIdx);
          if (got_j == hash[j].end()) {
            FeaturePoint feature_point;
            cv::Point2f pos = keypoints[j][matches[n].trainIdx].pt;
            feature_point.pos = Eigen::Vector2d(pos.x, pos.y);
            cv::Vec3i rgb = image_seq[j].get()->at<cv::Vec3b>(round(feature_point.pos[1]), round(feature_point.pos[0]));
            feature_point.rgb = Eigen::Vector3i(rgb[2], rgb[1], rgb[0]);
            feature_struct.feature_point.push_back(feature_point);
            hash[j].insert({matches[n].trainIdx, feature_struct.feature_point.size()});
          }
          int idx_j = feature_struct.feature_point.size();
          if(got_i == hash[i].end() && got_j == hash[j].end()) {
            triplet.push_back(Eigen::Triplet<int>(i, count_point, idx_i));
            triplet.push_back(Eigen::Triplet<int>(j, count_point, idx_j));
            count_point++;
          }
          else if(got_i == hash[i].end() && got_j != hash[j].end()) {
            //
          }
          else if(got_i != hash[i].end() && got_j == hash[j].end()) {
            //
          }
*/
          }
        }
        feature_struct.skeleton(i, j) = count;
        feature_struct.skeleton(j, i) = count;
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
      this_match.push_back(this_this_match);
    }

  }

//  for(int n = 0; n < triplet.size(); n++)
//    std::cout << triplet[n].row() << ' ' << triplet[n].col() << ' ' << triplet[n].value() << std::endl;
 
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
