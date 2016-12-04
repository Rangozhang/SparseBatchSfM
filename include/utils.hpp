#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <memory>
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(forward<Args>()(args)...));
}

bool arg_parser(int argc, char** argv, const char* arg_keys,
                    string* input_path);

#endif
