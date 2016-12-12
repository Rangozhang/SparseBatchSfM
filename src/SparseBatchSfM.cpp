#ifndef SPARSEBATCHSFM_CPP_
#define SPARSEBATCHSFM_CPP_

/* Awesome reference for Eigen
 * https://eigen.tuxfamily.org/dox/AsciiQuickReference.txt
 */

#include <fstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <time.h>

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
    // Making sure that j > i
    for (int i = 0; i < len; ++i) {
      for (int j = i+1; j < len; ++j) {
        if (!sk(i, j)) continue;
        edges.push_back(Edge(i, j, sk(i, j)));
      }
    }
    // for (int i = 0; i < len-1; ++i) {
    // for (int i = len-2; i >= 0; --i) {
    //   std::cout << i << ' ' << i+1 << ' ' << sk(i, i+1) << std::endl;
    //   edges.push_back(Edge(i, i+1, sk(i, i+1)));
    // }
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
    graph_merge_.reset(new GraphMerge());
  }

  SparseBatchSfM::~SparseBatchSfM() {
    image_capture_.reset();
    feature_processor_.reset();
    twoview_reconstruction_.reset();
    graph_merge_.reset(new GraphMerge());
  }

  SparseBatchSfM* SparseBatchSfM::instance_ = nullptr;

  SparseBatchSfM* SparseBatchSfM::getInstance() {
    if (!instance_) {
      instance_ = new SparseBatchSfM();
    }
    return instance_;
  }

  bool SparseBatchSfM::writeGraphToPLYFile(const GraphStruct& graph,
                                           const char* filename) {
    std::ofstream of(filename);

    int n_points = graph.Str.cols();

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
      of << graph.Str(0, i) << ' ' << graph.Str(1, i) << ' ' << graph.Str(2, i) << ' '
         << graph.Str(3, i) << ' ' << graph.Str(4, i) << ' ' << graph.Str(5, i) << '\n';
    }

    of.close();

    return true;
  }

  bool SparseBatchSfM::writeGraphToPLYFile(const GraphStruct& graph,
                                           std::unordered_map<int, int> hash,
                                           const char* filename) {
    std::ofstream of(filename);

    int n_points = graph.Str.cols();

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << n_points
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "element edge " << n_points - 1
       << '\n' << "property int vertex1"
       << '\n' << "property int vertex2"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    for (int i = 0; i < n_points; ++i) {
      of << graph.Str(0, i) << ' ' << graph.Str(1, i) << ' ' << graph.Str(2, i) << ' '
         << graph.Str(3, i) << ' ' << graph.Str(4, i) << ' ' << graph.Str(5, i) << '\n';
    }

    of << 0 << ' ' << 1 << ' ' << graph.Str(3, 1) << ' ' << graph.Str(4, 1) << ' ' << graph.Str(5, 1) << '\n';
    for (int i = 2; i < n_points; ++i) {
      of << hash[i + 1] << ' ' << i << ' ' << graph.Str(3, i) << ' ' << graph.Str(4, i) << ' ' << graph.Str(5, i) << '\n';
    }

    of.close();

    return true;
}

  void SparseBatchSfM::run(const std::string& input_path) {
    clock_t t1 = clock(), t2, t3;
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
    std::cout << "skeleton: " << std::endl << controller->feature_struct_.skeleton << std::endl;
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
    controller->feature_processor_->skeletonize(controller->feature_struct_.skeleton, 140);

    std::cout << "skeleton: " << std::endl << controller->feature_struct_.skeleton << std::endl;
    std::vector<Edge> edges = {};
    convertToVectors(controller->feature_struct_.skeleton, edges);
    if (!edges.size()) {
      std::cout << "edges size 0" << std::endl;
      return;
    }
    // return;
    // for (const auto& edge : edges) {
    //     std::cout << edge.idx1 << ' ' << edge.idx2 << ' ' << edge.weight << std::endl;
    // }

    /****** Twoview Reconstruction along the skeleton ******/
    std::cout << "Two view Reconstruction" << std::endl;
    // std::vector<Edge> tmp_edges = {edges[0]};
    for (const auto& edge : edges) {
        Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
        K1 << 1520.4, 0, 302.32, 0, 1525.9, 246.87, 0, 0, 1;
        K2 << 1520.4, 0, 302.32, 0, 1525.9, 246.87, 0, 0, 1;
        // twoview reconstruction for each edge
        std::unique_ptr<GraphStruct> graph;
        graph.reset(new GraphStruct());
        std::cout << "Reconstruct " << edge.idx1 << " " << edge.idx2 << "..." << std::endl;
        if (!controller->twoview_reconstruction_->reconstruct(controller->feature_struct_,
                                                         edge.idx1, edge.idx2, img_width, img_height,
                                                         K1, K2, *graph.get(), *controller->image_seq_[edge.idx1].get(), *controller->image_seq_[edge.idx2].get())) {
            std::cerr << "Failed to twoview reconstruct" << std::endl;
            return;
        }

        // std::cout << "Structure: " << std::endl;
        // for (int i = 0; i < graph->Str.cols(); ++i) {
        //   std::cout << graph->Str(0, i) << ' ' << graph->Str(1, i) << ' ' << graph->Str(2, i) << std::endl;
        // }

        // BundleAdjustment
        std::cout << "BundleAdjustment" <<std::endl;
        // std::cout << graph->Str(0, 0) << ' ' << graph->Str(1, 0)
	    //  		 << ' ' << graph->Str(2, 0) << std::endl;
        // std::cout << graph->Mot[0] << std::endl;

        BundleAdjustment ba;
        ba.run(*graph.get());

        std::string two_view_file = "output/TwoView_" + std::to_string(edge.idx1) + "_" + std::to_string(edge.idx2) + ".ply";
        controller->writeGraphToPLYFile(*graph.get(), two_view_file.c_str());


        // std::cout << graph->Str(0, 0) << ' ' << graph->Str(1, 0)
	    //  		 << ' ' << graph->Str(2, 0) << std::endl;
        // std::cout << graph->Mot[0] << std::endl;

        // std::cout << graph->feature_idx << std::endl;

        controller->graphs_.push_back(std::move(graph));
    }
    t2 = clock();

    /****** Merge graphs ******/
    std::cout << "Merge Graphs" << std::endl;
    std::unordered_set<int> visited_frames;
    for (const auto& idx : controller->graphs_[0]->frame_idx) {
      visited_frames.insert(idx);
    }

    int merge_count = 0;
    std::unordered_map<int, int> curind_preind = {};
    while (controller->graphs_.size() > 1) {
      int ind = 1;
      // The order of the graph array has been sorted according to matches
      for (int i = 1; i < controller->graphs_.size(); ++i) {
        if (hasIdxInHashSet(controller->graphs_[i]->frame_idx, visited_frames)) {
          /*std::cout << "Merging Graph with frame ";
          for (int frame : controller->graphs_[i]->frame_idx) {
            std::cout << std::to_string(frame) << ' ';
          }
          std::cout << std::endl;*/
          ind = i;
          break;
        }
      }

      // Merge two graphs
      if (!controller->graph_merge_->merge(*controller->graphs_[0].get(), *controller->graphs_[ind].get())) {
        std::cout << "Merging failed" << std::endl;
        return;
      }

      if (!controller->graph_merge_->multiTriangulate(*controller->graphs_[0].get())) {
        std::cout << "MultiTriangulate failed" << std::endl;
        return;
      }
      // std::cout << "Multi-triangulate: " << std::endl;
      // std::cout << (*controller->graphs_[0].get()).Str.leftCols(5);

      std::cout << "BundleAdjustment" <<std::endl;
      BundleAdjustment ba;
      ba.run(*controller->graphs_[0].get());

      std::string tmp_file = "output/tmp_merge_" + std::to_string(merge_count++) + ".ply";
      controller->writeGraphToPLYFile(*controller->graphs_[0].get(), tmp_file.c_str());

      // put the new vertex in to hash set
      int pre = 0;
      for (const auto& frame_idx : controller->graphs_[ind]->frame_idx) {
        if (visited_frames.count(frame_idx)) {
          pre = frame_idx;
        }
        visited_frames.insert(frame_idx);
      }
      for (int i = 0; i < controller->graphs_[0]->frame_idx.size(); ++i) {
        if (controller->graphs_[0]->frame_idx[i] == pre) {
          curind_preind[controller->graphs_[0]->frame_idx.size()] = i;
          break;
        }
      }

      controller->graphs_.erase(controller->graphs_.begin() + ind);
    }

    GraphStruct tmp_graph;
    tmp_graph.Str.resize(6, seq_len);
    std::cout << "Frame merge order: ";
    for (int i = 0; i < seq_len; ++i) {
      std::cout << controller->graphs_[0]->frame_idx[i] << " ";
      Eigen::MatrixXd K = controller->graphs_[0]->K[i];
      Eigen::MatrixXd M = K * controller->graphs_[0]->Mot[i];
      tmp_graph.Str.block(0, i, 3, 1) = -M.leftCols(3).inverse()*M.rightCols(1);
      if (i == 0 || i == 1) {
        tmp_graph.Str.block(3, i, 3, 1) << 255, 0, 0;
      } else {
        tmp_graph.Str.block(3, i, 3, 1) << 0, 255 - 10*(i-2), 0;
      }
    }
    std::cout << std::endl;
    if (!controller->writeGraphToPLYFile(tmp_graph, curind_preind,  "./output/camera_pos.ply")) {
      std::cerr << "Can not write the camera pos to .ply file";
    }

    if (!controller->writeGraphToPLYFile(*controller->graphs_[0].get(), "./output/result.ply")) {
      std::cerr << "Can not write the structure to .ply file.";
    }

    t3 = clock();
    std::cout << "Time before merging: " << (float(t2) - float(t1)) / CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Time of merging: " << (float(t3) - float(t2)) / CLOCKS_PER_SEC << " seconds" << std::endl;

    return;
  }

} // namespace sparse_batch_sfm

#endif
