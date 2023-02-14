#include "mot_eval.hpp"

int main(int argc, char**argv) {
  std::string input_dets_loc = argv[1];
  float track_thresh = atof(argv[2]);
  std::cout << "[ INFO ] Input detections: " << input_dets_loc << std::endl;

  for (auto const &dir_entry : std::filesystem::directory_iterator{input_dets_loc}) {
    dets_map_t mot_dets = LoadMotDetections(dir_entry.path());

    // Create output stream for mot formatted tracks
    const std::string output_mot_file = dir_entry.path().filename();
    std::ofstream output_txt_file(output_mot_file, std::ios_base::app);

    // Initialize new bytetracker for each MOT video
    int frame_rate   = 30;
    int track_buffer = 30;
    float min_box_area = 20;
    float aspect_ratio_thresh = 1.6;

    // Determine if current mot vdeo has a custom config
    std::string mot_name = dir_entry.path().filename().replace_extension();
    if (MOT_CONFIG.count(mot_name) != 0) {
      track_buffer = MOT_CONFIG[mot_name].first;
      track_thresh = MOT_CONFIG[mot_name].second;
    }

    bt::BYTETracker tracker(frame_rate, track_buffer, track_thresh);

    int frame_id = 1;
    int total_frames = mot_dets.size();

    PrintConfig(mot_name, total_frames, track_thresh, track_buffer);

    while (frame_id <= total_frames) {
      // Update bytetracker and retrieve tracks
      std::vector<bt::STrack> tracks = tracker.update(mot_dets[frame_id]);

      // Filter tracks based on box aspect ratio and min size
      for (const auto &track : tracks) {
        std::vector<float> tlwh = track.tlwh;
        bool vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh;
        if (tlwh[2] * tlwh[3] > min_box_area && !vertical) {
          // Valid bbox, write to text file
          if (output_txt_file.is_open()) {
            output_txt_file << std::to_string(frame_id) << ","
                            << std::to_string(track.track_id) << ","
                            << tlwh[0] << "," << tlwh[1] << ","
                            << tlwh[2] << "," << tlwh[3] << ","
                            << track.score << ",-1,-1,-1\n";
          } else {
            std::cout << "[ ERROR ] Cannot open: " << output_mot_file << std::endl;
          }
        }
      }

      frame_id++;
    }

    output_txt_file.close();
  }

  return 0;
}

