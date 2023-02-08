#ifndef BYTETRACK_MOT_GENERATE_TRACKS_HPP_
#define BYTETRACK_MOT_GENERATE_TRACKS_HPP_

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <bytetrack/BYTETracker.h>

namespace bt = bytetrack;

typedef std::map<int, std::vector<bt::Object>> dets_map_t;

dets_map_t LoadMotDetections(const std::string &det_loc) {
  std::cout << "[ INFO ] Loading detection for: " << det_loc << std::endl;

  dets_map_t detections;

  std::ifstream gt_file;
  gt_file.open(det_loc);

  if (gt_file.is_open()) {
    while (gt_file.good()) {

      std::string s, line;
      if (!std::getline( gt_file, line )) break;
      std::istringstream ss(line);

      if (!std::getline( ss, s, ',' )) break;
      int frame_id = std::stoi(s);

      if (!std::getline( ss, s, ',' )) break;
      int det_id = std::stoi(s);

      if (!std::getline( ss, s, ',' )) break;
      double bb_left = std::stod(s);

      if (!std::getline( ss, s, ',' )) break;
      double bb_top = std::stod(s);

      if (!std::getline( ss, s, ',' )) break;
      double bb_width = std::stod(s);

      if (!std::getline( ss, s, ',' )) break;
      double bb_height = std::stod(s);

      if (!std::getline( ss, s, ',' )) break;
      float conf = std::stof(s);

      if (!std::getline( ss, s, ',' )) break;
      double label = std::stod(s);

      // Create bytetracker detection object
      bt::Object obj;
      obj.rect.x = bb_left;
      obj.rect.y = bb_top;
      obj.rect.width  = bb_width;
      obj.rect.height = bb_height;
      obj.prob  = conf;
      obj.label = label;

      detections[frame_id].push_back(obj);
    }
    gt_file.close();
  } else {
    std::cout << "[ ERROR ] Cannot open file: " << det_loc << std::endl;
  }

  return detections;
}

#endif  // BYTETRACK_MOT_GENERATE_TRACKS_HPP_
