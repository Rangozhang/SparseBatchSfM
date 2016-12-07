#ifndef SPARSEBATCHSFM_CPP_
#define SPARSEBATCHSFM_CPP_

#include <fstream>
#include <unordered_set>

#include "SparseBatchSfM.hpp"

namespace sparse_batch_sfm {
namespace {
  struct Edge {
    int idx1, idx2;
    int weight;
    Edge(int idx1, int idx2, int weight):idx1(idx1), idx2(idx2), weight(weight) {};
  };
  void convertToVectors(const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& sk,
                        std::vector<Edge>& edges) {
    int len = sk.rows();
    for (int i = 0; i < len; ++i) {
      for (int j = i+1; j < len; ++j) {
        if (!sk(i, j)) continue;
        edges.push_back(Edge(i, j, sk(i, j)));
      }
    }
    std::sort(edges.begin(), edges.end(),
            [](const Edge& edge1, const Edge& edge2){return edge1.weight > edge2.weight;});
    return;
  }
  
  bool hasIdxInHashSet(const std::vector<int>& frame_idx,
                       const std::unordered_set<int>& visited_frames) {
    for (const auto& frame : frame_idx) {
      if (visited_frames.count(frame)) {
        return true;
      }
    }
    return false;
  }
}

  SparseBatchSfM::SparseBatchSfM() {
    image_capture_.reset(new ImageCapture());
    feature_processor_.reset(new FeatureProcessor());
    twoview_reconstruction_.reset(new TwoViewReconstruction());
  }

  SparseBatchSfM::~SparseBatchSfM() {
    image_capture_.reset(); 
    feature_processor_.reset();
    twoview_reconstruction_.reset();
  }

  SparseBatchSfM* SparseBatchSfM::instance_ = nullptr;

  SparseBatchSfM* SparseBatchSfM::getInstance() {
    if (!instance_) {
      instance_ = new SparseBatchSfM();
    }
    return instance_;
  }

  bool SparseBatchSfM::writeGraphToPLYFile(const std::vector<std::unique_ptr<GraphStruct>>& graphs,
                                           const char* filename) {
    // only write down the first graph in graphs
    if (!graphs.size()) return false;
    const std::unique_ptr<GraphStruct>& graph = graphs[0];

    std::ofstream of(filename);

    int n_points = graph->Str.cols();

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << n_points
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    for (int i = 0; i < n_points; ++i) {
      of << graph->Str(0, i) << ' ' << graph->Str(1, i) << ' ' << graph->Str(2, i)
         << graph->Str(3, i) << ' ' << graph->Str(4, i) << ' ' << graph->Str(5, i) << '\n';
    }

    of.close();
    
    return true;
  }

  void SparseBatchSfM::run(const std::string& input_path) {
    std::cout << "INPUT PARAMS" << std::endl;
    std::cout << "Input path: " << input_path << std::endl;
    SparseBatchSfM* controller = controller->getInstance();

    /************** Read images from Dir ***************/
    if (!controller->image_capture_->ReadFromDir(
                  input_path, controller->image_seq_)) {
      return;
    }

    int seq_len = controller->image_seq_.size();
    int img_width = controller->image_seq_[0]->cols;
    int img_height = controller->image_seq_[0]->rows;
    
    /************** Processing feature ***************/
    std::cout << "Feature processing" << std::endl;
    controller->feature_processor_->feature_match(controller->image_seq_, controller->feature_struct_, 400, 200, false);
    /*
    for(int i = 0; i < seq_len; i++) {
      std::cout << controller->feature_struct_.feature_point[i].size() << std::endl;
      std::cout << controller->feature_struct_.feature_point[i][0].pos << std::endl;
      for(int j = 0; j < seq_len; j++) {
        std::cout << controller->feature_struct_.feature_matches[i][j].size() << std::endl;
      }
    }
    */
    // std::cout << "skeleton: " << std::endl << controller->feature_struct_.skeleton << std::endl;
    // controller->feature_struct_.skeleton.resize(8, 8);
    // controller->feature_struct_.skeleton << 0, 1, 0, 10, 0, 0, 0, 0,
    //                                         1, 0, 10, 0, 3, 0, 0, 0,
    //                                         0, 10, 0, 4, 9, 0, 0, 0,
    //                                         10, 0, 4, 0, 0, 7, 0, 0,
    //                                         0,  3, 9, 0, 0, 6, 0, 0,
    //                                         0,  0, 0, 7, 6, 0, 7, 8,
    //                                         0,  0, 0, 0, 0, 7, 0, 5,
    //                                         0,  0, 0, 0, 0, 8, 5, 0;
    // controller->feature_processor_->skeletonize(controller->feature_struct_.skeleton, 0);
    // std::cout << "skeleton: " << std::endl << controller->feature_struct_.skeleton << std::endl;
    controller->feature_processor_->skeletonize(controller->feature_struct_.skeleton, 20);

    std::vector<Edge> edges = {};
    convertToVectors(controller->feature_struct_.skeleton, edges);
    if (!edges.size()) {
      return;
    }
    // for (const auto& edge : edges) {
    //     std::cout << edge.idx1 << ' ' << edge.idx2 << ' ' << edge.weight << std::endl;
    // }

    /****** Twoview Reconstruction along the skeleton ******/
    std::cout << "Two view Reconstruction" << std::endl;
    for (const auto& edge : edges) {
        // Get intrinsic matrix TODO: Change to real K
        Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
        // twoview reconstruction for each edge
        std::unique_ptr<GraphStruct> graph;
        graph.reset(new GraphStruct());
        if (!controller->twoview_reconstruction_->reconstruct(controller->feature_struct_,
                                                         edge.idx1, edge.idx2, img_width, img_height,
                                                         K1, K2, *graph.get(), *controller->image_seq_[edge.idx1].get(), *controller->image_seq_[edge.idx2].get())) {
            std::cerr << "Failed to twoview reconstruct" << std::endl;
            return;
        }

        // BundleAdjustment
        
        controller->graphs_.push_back(std::move(graph));
    }
    

    /****** Merge graphs ******/
    std::cout << "Merge Graphs" << std::endl;
    std::unordered_set<int> visited_frames;
    for (const auto& idx : controller->graphs_[0]->frame_idx) {
      visited_frames.insert(idx);
    }
    while (controller->graphs_.size() > 1) {
      int ind = 1;
      for (int i = 1; i < controller->graphs_.size(); ++i) {
        if (hasIdxInHashSet(controller->graphs_[i]->frame_idx, visited_frames)) {
          ind = i;
          break;
        }
      }

      // merge the second one to the first one
      // merge(graphs_[0], graphs_[ind]);

      // bundleadjustment(graphs_[0]);
      
      // put the new vertex in to hash set
      for (const auto& frame_idx : controller->graphs_[ind]->frame_idx) {
        visited_frames.insert(frame_idx);
      }

      controller->graphs_.erase(controller->graphs_.begin() + ind);
    }

    if (!controller->writeGraphToPLYFile(controller->graphs_, "./result.ply")) {
      std::cerr << "Can not write the structure to .ply file.";
    }

    return;
  }

} // namespace sparse_batch_sfm

#endif
