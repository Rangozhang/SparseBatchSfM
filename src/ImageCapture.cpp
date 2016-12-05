#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImageCapture.hpp"

namespace sparse_batch_sfm {
  bool ImageCapture::ReadFromDir(const std::string& input_path,
                   std::vector<std::unique_ptr<cv::Mat>>& image_seq) {

    cv::VideoCapture sequence (input_path + "/%04d.jpg");
    bool show_im = false;

    if (!sequence.isOpened()) {
      std::cerr << "Failed to open Image Sequence!\n" << std::endl;
      return false;
    }

    if (show_im) {
      cv::namedWindow("Input preview", CV_WINDOW_NORMAL);
    }
    std::unique_ptr<cv::Mat> tmp_img;

    for(;;) {
      tmp_img.reset(new cv::Mat());
      sequence >> *(tmp_img.get());
      if(tmp_img->empty()) {
        if (image_seq.size() == 0) {
          std::cerr << "There is no images in the given directory";
          return false;
        } else {
          std::cout << "Finish reading all "
               << image_seq.size() << " images" << std::endl;
        }
        break;
      }

      image_seq.push_back(make_unique<cv::Mat>());
      image_seq.back().reset(tmp_img.release());
      if (show_im) {
        cv::imshow("Input preview", *image_seq.back().get());
        char key = (char)cv::waitKey(10);
        if (key == 'q' || key == 27) {
          show_im = false;
        }
      }
    }

    return true;
  }
} // namespace sparse_batch_sfm
