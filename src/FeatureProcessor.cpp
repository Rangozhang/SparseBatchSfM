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
  }

  // match features between each pair of images
  std::vector<std::unordered_map<int, int>> hash(seq_len);
  std::vector<Eigen::Triplet<int>> triplet;
  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;
  int count_point = 0;
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

  }

  for(int n = 0; n < triplet.size(); n++)
    std::cout << triplet[n].row() << ' ' << triplet[n].col() << ' ' << triplet[n].value() << std::endl;
 
  return true;
  }

  bool FeatureProcessor::skeletonize(Eigen::Matrix<int, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>& skeleton) {

  }
}
