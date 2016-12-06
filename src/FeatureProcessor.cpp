#include <unordered_set>

#include "FeatureProcessor.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"

namespace sparse_batch_sfm {
  bool FeatureProcessor::feature_match(const std::vector<std::unique_ptr<cv::Mat>>& image_seq, FeatureStruct& feature_struct) {
  
  // SIFT detector and extractor
  int minHessian = 400;
  cv::SiftFeatureDetector detector(minHessian);
  cv::SiftDescriptorExtractor extractor;
  
  // detect keypoints and extract features for all images
  int seq_len = image_seq.size();
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
  float thres = 200;
  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;
  for (int i = 0; i < seq_len; i++) {
    for (int j = i + 1; j < seq_len; j++) {
      matcher.match(descriptors[i], descriptors[j], matches);
      cv::Mat img_matches;
      cv::drawMatches(*image_seq[i].get(), keypoints[i], *image_seq[j].get(), keypoints[j],
               matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
      cv::imshow("Matches", img_matches);
      cv::waitKey(0);
    }
  }
  
  return true;
  }

  bool FeatureProcessor::skeletonize(Eigen::Matrix<int, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>& skeleton) {
    
  }
}
