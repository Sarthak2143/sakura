#include "sakura.hpp"
#include <cpr/cpr.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>
#include <getopt.h>

// Helper to get terminal pixel size (returns {width, height} in pixels)
std::pair<int, int> getTerminalPixelSize() {
  struct winsize w;
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_xpixel > 0 &&
      w.ws_ypixel > 0) {
    return {w.ws_xpixel, w.ws_ypixel};
  }
  // Fallback: return a reasonable default if pixel size is not available
  return {1920, 1080};
}

// Helper to calculate best-fit dimensions for content
std::pair<int, int> calculateBestFitSize(int contentWidth, int contentHeight,
                                         int termWidth, int termHeight) {
  double contentAspect = static_cast<double>(contentWidth) / contentHeight;
  double termAspect = static_cast<double>(termWidth) / termHeight;

  int outw, outh;
  if (contentAspect > termAspect) {
    // Content is wider - fit to terminal width
    outw = termWidth;
    outh = static_cast<int>(termWidth / contentAspect);
  } else {
    // Content is taller - fit to terminal height
    outh = termHeight;
    outw = static_cast<int>(termHeight * contentAspect);
  }
  return {outw, outh};
}

bool process_image(std::string url) {
  Sakura sakura;
  bool stat = false;
  auto [termPixW, termPixH] = getTerminalPixelSize();

  auto response = cpr::Get(cpr::Url{url});
  if (response.status_code != 200) {
    std::cerr << "Failed to download image. Status: " << response.status_code
              << std::endl;
    return 1;
  }
  std::vector<uchar> imgData(response.text.begin(), response.text.end());
  cv::Mat img = cv::imdecode(imgData, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Failed to decode image" << std::endl;
    return 1;
  }

  // Calculate proper dimensions for this image
  auto [outw, outh] =
      calculateBestFitSize(img.cols, img.rows, termPixW, termPixH);

  Sakura::RenderOptions options;
  options.mode = Sakura::SIXEL;
  options.dither = Sakura::FLOYD_STEINBERG;
  options.terminalAspectRatio = 1.0;
  options.width = outw;
  options.height = outh;

  return sakura.renderFromMat(img, options);
}

bool process_gif(std::string url) {
  Sakura sakura;
  bool stat = false;
  auto [termPixW, termPixH] = getTerminalPixelSize();

  // For GIF, we'll let the sakura library handle sizing internally
  // but still set reasonable defaults
  Sakura::RenderOptions options;
  options.mode = Sakura::SIXEL;
  options.dither = Sakura::FLOYD_STEINBERG;
  options.terminalAspectRatio = 1.0;
  options.width = termPixW;
  options.height = termPixH;

  return sakura.renderGifFromUrl(url, options);
}

bool process_video(std::string url) {
  Sakura sakura;
  bool stat = false;
  auto [termPixW, termPixH] = getTerminalPixelSize();

  // For video, we'll let the sakura library handle sizing internally
  // but still set reasonable defaults
  Sakura::RenderOptions options;
  options.mode = Sakura::SIXEL;
  options.dither = Sakura::FLOYD_STEINBERG;
  options.terminalAspectRatio = 1.0;
  options.width = termPixW;
  options.height = termPixH;

  return sakura.renderVideoFromUrl(url, options);
}

bool process_local_video(std::string path) {
  Sakura sakura;
  bool stat = false;
  auto [termPixW, termPixH] = getTerminalPixelSize();

  // For video, we'll let the sakura library handle sizing internally
  // but still set reasonable defaults
  Sakura::RenderOptions options;
  options.mode = Sakura::SIXEL;
  options.dither = Sakura::FLOYD_STEINBERG;
  options.terminalAspectRatio = 1.0;
  options.width = termPixW;
  options.height = termPixH;

  stat = sakura.renderVideoFromFile(path, options);
  return stat;
}

int main(int argc, char **argv) {
  // Parse command line arguments
  static struct option long_options[] = {
      {"help",        no_argument,       0, 'h'},
      {"image",       required_argument, 0, 'i'},
      {"gif",         required_argument, 0, 'g'},
      {"video",       required_argument, 0, 'v'},
      {"local-video", required_argument, 0, 'l'},
      {0, 0, 0, 0}
  };
  
  std::string video_path, image_path;
  bool show_help = false;
  
  int opt;
  int option_index = 0;
  bool stat = false;

  if (argc > 1) {
    while ((opt = getopt_long(argc, argv, "hv:i:g:l:", long_options, &option_index)) != -1) {
      switch (opt) {
        case 'h':
          std::cout << "Usage: sakura [options]\n"
                    << "Options:\n"
                    << "  -h, --help                 Show help message\n"
                    << "  -i, --image <path>         Process image file\n"
                    << "  -g, --gif <path>           Process GIF file\n"
                    << "  -v, --video <path>         Process video file\n"
                    << "  -l, --local-video <path>   Process local video file\n"
          ;
          return 0;
                  
        case 'i':
          stat = process_image( optarg );
          break;
        
        case 'g':
          stat = process_gif( optarg );
          break;
        
        case 'v':
          stat = process_video( optarg );
          break;
        
        case 'l':
          stat = process_local_video( optarg );
          break;
          
        case '?':
          // getopt_long automatically prints error message
          return 1;
          
        default:
          std::cerr << "Unknown option: " << opt << std::endl;
          return 1;
      }
    }
    if (!stat) {
      std::cerr << "Failed to render content\n";
    }
    return 0;
  }
	
	// Handle any remaining non-option arguments
	for (int i = optind; i < argc; i++) {
		std::cout << "Non-option argument: " << argv[i] << std::endl;
	}

  // Continue to existing interactive mode
  Sakura sakura;

  std::string url, gifUrl, videoUrl, localVideoPath;

  // Get terminal pixel size once
  auto [termPixW, termPixH] = getTerminalPixelSize();

  std::cout << "Sakura Video Player with SIXEL\n";
  std::cout << "1. Image\n2. GIF\n3. Video (URL)\n4. Video (File)\n";
  std::cout << "Choose option (1-4): ";

  int choice;
  std::cin >> choice;

  switch (choice) {
  case 1: {
    std::cout << "Enter image URL: ";
    std::cin >> url;
    std::cout << "Rendering image...\n";
    stat = process_image(url);
    break;
  }
  case 2: {
    std::cout << "Enter GIF URL: ";
    std::cin >> gifUrl;
    std::cout << "Rendering GIF...\n";
    stat = process_gif(gifUrl);
    break;
  }
  case 3: {
    std::cout << "Enter video URL: ";
    std::cin >> videoUrl;
    std::cout << "Rendering video from URL (with audio)...\n";
    stat = process_video(videoUrl);
    break;
  }
  case 4: {
    std::cout << "Enter video file path: ";
    std::cin >> localVideoPath;
    std::cout << "Rendering video from file (with audio)...\n";

    stat = process_local_video(localVideoPath);
    break;
  }
  default: {
    std::cout << "Invalid choice. Rendering image by default.\n";
    std::cout << "Enter image URL: ";
    std::cin >> url;
    std::cout << "Rendering image...\n";
    // Download image and calculate its dimensions
    auto response = cpr::Get(cpr::Url{url});
    if (response.status_code != 200) {
      std::cerr << "Failed to download image. Status: " << response.status_code
                << std::endl;
      return 1;
    }
    std::vector<uchar> imgData(response.text.begin(), response.text.end());
    cv::Mat img = cv::imdecode(imgData, cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cerr << "Failed to decode image" << std::endl;
      return 1;
    }

    auto [outw, outh] =
        calculateBestFitSize(img.cols, img.rows, termPixW, termPixH);

    Sakura::RenderOptions options;
    options.mode = Sakura::SIXEL;
    options.dither = Sakura::FLOYD_STEINBERG;
    options.terminalAspectRatio = 1.0;
    options.width = outw;
    options.height = outh;

    stat = sakura.renderFromMat(img, options);
    break;
  }
  }

  if (!stat) {
    std::cerr << "Failed to render content\n";
  }
  return 0;
}
