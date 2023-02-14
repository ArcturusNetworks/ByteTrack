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

void PrintConfig(const std::string &name, const int &num_frames,
    const float &track_thresh, const int &track_buffer) {
  std::cout << "[ INFO ] ByteTrack Config for " << name << " ... " << std::endl;
  std::cout << "[ INFO ]     Total Frames: " << num_frames << std::endl;
  std::cout << "[ INFO ]     Track Thresh: " << track_thresh << std::endl;
  std::cout << "[ INFO ]     Track Buffer: " << track_buffer << std::endl;
}

// Define unique tracking configurations for different MOT videos
// std::pair<track_buffer, track_thresh>
std::unordered_map<std::string, std::pair<int, float>> MOT_CONFIG = {
  {"MOT17-01-FRCNN", {30, 0.65}}, {"MOT17-05-FRCNN", {14, 0.50}},
  {"MOT17-06-FRCNN", {14, 0.65}}, {"MOT17-12-FRCNN", {30, 0.70}},
  {"MOT17-13-FRCNN", {25, 0.50}}, {"MOT17-14-FRCNN", {25, 0.67}},
  {"MOT20-06", {30, 0.30}}, {"MOT20-08", {30, 0.30}}
};

#endif  // BYTETRACK_MOT_GENERATE_TRACKS_HPP_
