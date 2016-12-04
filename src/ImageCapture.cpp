#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImageCapture.hpp"

namespace sparse_batch_sfm {
  bool ImageCapture::ReadFromDir(const string& input_path,
                   vector<unique_ptr<Mat>>& image_seq) {

    VideoCapture sequence (input_path + "/%04d.jpg");
    bool show_im = false;

    if (!sequence.isOpened()) {
      cerr << "Failed to open Image Sequence!\n" << endl;
      return false;
    }

    if (show_im) {
      namedWindow("Input preview", CV_WINDOW_NORMAL);
    }
    unique_ptr<Mat> tmp_img;

    for(;;) {
      tmp_img.reset(new Mat());
      sequence >> *(tmp_img.get());
      if(tmp_img->empty()) {
        if (image_seq.size() == 0) {
          cerr << "There is no images in the given directory";
          return false;
        } else {
          cout << "Finish reading all "
               << image_seq.size() << " images" << endl;
        }
        break;
      }

      image_seq.push_back(make_unique<Mat>());
      image_seq.back().reset(tmp_img.release());
      if (show_im) {
        imshow("Input preview", *image_seq.back().get());
        char key = (char)waitKey(10);
        if (key == 'q' || key == 27) {
          show_im = false;
        }
      }
    }

    return true;
  }
} // namespace sparse_batch_sfm
