#include "sakura.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cpr/cpr.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sixel.h>
#include <sstream>
#include <string_view>
#include <thread>
#include <vector>

const std::string Sakura::ASCII_CHARS_SIMPLE = " .:-=+*#%@";
const std::string Sakura::ASCII_CHARS_DETAILED =
    " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
const std::string Sakura::ASCII_CHARS_BLOCKS = " \u2591\u2592\u2593\u2588";

#ifdef _WIN32
#include <windows.h>
std::pair<int, int> Sakura::getTerminalSize() {
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
  return {csbi.srWindow.Right - csbi.srWindow.Left + 1,
          csbi.srWindow.Bottom - csbi.srWindow.Top + 1};
}
#else
#include <sys/ioctl.h>
#include <unistd.h>
std::pair<int, int> Sakura::getTerminalSize() {
  struct winsize w;
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
    return {w.ws_col, w.ws_row};
  }
  return {80, 24};
}
#endif

bool Sakura::preprocessAndResize(const cv::Mat &img,
                                 const RenderOptions &options, cv::Mat &resized,
                                 int &target_width, int &target_height) const {
  cv::Mat adjusted;
  if (options.contrast != 1.0 || options.brightness != 0.0) {
    img.convertTo(adjusted, -1, options.contrast * 1.2, options.brightness);
  } else {
    adjusted = img;
  }

  target_width = options.width;
  target_height = options.height;

  if (target_width == 0 || target_height == 0) {
    const auto [w, h] = getTerminalSize();
    if (target_width == 0)
      target_width = w;
    if (target_height == 0)
      target_height = h;
  }

  if (options.aspectRatio) {
    double aspectRatio = static_cast<double>(adjusted.cols) / adjusted.rows;
    if (options.mode == EXACT || options.mode == ASCII_COLOR ||
        options.mode == SIXEL) {
      aspectRatio /= options.terminalAspectRatio;
    }

    if (aspectRatio > static_cast<double>(target_width) / target_height) {
      target_height = static_cast<int>(target_width / aspectRatio);
    } else {
      target_width = static_cast<int>(target_height * aspectRatio);
    }
    target_width = std::max(target_width, 1);
    target_height = std::max(target_height, 1);
  }

  const cv::Size targetSize = (options.mode == EXACT)
                                  ? cv::Size(target_width, target_height * 2)
                                  : cv::Size(target_width, target_height);

  cv::resize(adjusted, resized, targetSize, 0, 0, cv::INTER_AREA);
  return !resized.empty();
}

// Helper to get header and value out of headers, Ex: "header: value" -> header, value
std::pair<std::string, std::string> Sakura::splitHeader(const std::string &headerLine) {
    size_t pos = headerLine.find(':');
    if (pos == std::string::npos) {
        return {"", ""};
    }

    std::string header = headerLine.substr(0, pos);
    std::string value = headerLine.substr(pos + 1);

    auto trim = [](std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    };

    trim(header);
    trim(value);

    return {header, value};
}


bool Sakura::renderFromUrl(std::string_view url,
                           const RenderOptions &options) const {
  const auto response = cpr::Get(cpr::Url{std::string(url)});
  if (response.status_code != 200) {
    std::cerr << "Failed to download image. Status: " << response.status_code
              << std::endl;
    return false;
  }

  const std::vector<uchar> imgData(response.text.begin(), response.text.end());
  const cv::Mat img = cv::imdecode(imgData, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Failed to decode image" << std::endl;
    return false;
  }
  return renderFromMat(img, options);
}

bool Sakura::renderFromUrl(std::string_view url) const {
  RenderOptions options;
  options.mode = EXACT;
  return renderFromUrl(url, options);
}

bool Sakura::renderFromMat(const cv::Mat &img,
                           const RenderOptions &options) const {
  if (options.mode == SIXEL) {
    cv::Mat processed;
    if (img.cols != options.width || img.rows != options.height) {
      cv::resize(img, processed, cv::Size(options.width, options.height), 0, 0,
                 cv::INTER_NEAREST);
    } else {
      processed = img;
    }

    const std::string sixelData = renderSixel(processed, options.paletteSize);
    std::cout << sixelData << std::flush;
    return true;
  }

  cv::Mat resized;
  int target_width, target_height;
  if (!preprocessAndResize(img, options, resized, target_width,
                           target_height)) {
    std::cerr << "Resize failed" << std::endl;
    return false;
  }

  if ((options.mode == EXACT || options.mode == ASCII_COLOR) &&
      resized.channels() == 1) {
    cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
  }

  std::vector<std::string> lines;
  switch (options.mode) {
  case EXACT:
    lines = renderExact(resized, target_height);
    break;
  case ASCII_COLOR:
    lines = renderAsciiColor(resized);
    break;
  case ASCII_GRAY: {
    const std::string &charSet = getCharSet(options.style);
    lines = renderAsciiGrayscale(resized, charSet, options.dither);
    break;
  }
  default:
    return false;
  }

  for (const auto &line : lines) {
    std::cout << line << '\n';
  }
  std::cout.flush();
  return true;
}

const std::string &Sakura::getCharSet(CharStyle style) const noexcept {
  switch (style) {
  case SIMPLE:
    return ASCII_CHARS_SIMPLE;
  case DETAILED:
    return ASCII_CHARS_DETAILED;
  case BLOCKS:
    return ASCII_CHARS_BLOCKS;
  default:
    return ASCII_CHARS_DETAILED;
  }
}

std::vector<std::string> Sakura::renderExact(const cv::Mat &resized,
                                             int terminal_height) const {
  std::vector<std::string> lines;
  const int height = resized.rows / 2;
  const int width = resized.cols;
  const int max_lines = std::min(height, terminal_height);

  lines.reserve(max_lines);

  for (int k = 0; k < max_lines; ++k) {
    std::string line;
    line.reserve(width * 30); // Pre-allocate for ANSI sequences

    for (int j = 0; j < width; ++j) {
      const cv::Vec3b top_pixel = resized.at<cv::Vec3b>(2 * k, j);
      const cv::Vec3b bottom_pixel = (2 * k + 1 < resized.rows)
                                         ? resized.at<cv::Vec3b>(2 * k + 1, j)
                                         : top_pixel;

      // Use string stream for better performance with repeated formatting
      std::ostringstream oss;
      oss << "\x1b[48;2;" << static_cast<int>(bottom_pixel[2]) << ';'
          << static_cast<int>(bottom_pixel[1]) << ';'
          << static_cast<int>(bottom_pixel[0]) << "m\x1b[38;2;"
          << static_cast<int>(top_pixel[2]) << ';'
          << static_cast<int>(top_pixel[1]) << ';'
          << static_cast<int>(top_pixel[0]) << "m▀\x1b[0m";
      line += oss.str();
    }
    lines.emplace_back(std::move(line));
  }
  return lines;
}

std::vector<std::string>
Sakura::renderAsciiColor(const cv::Mat &resized) const {
  std::vector<std::string> lines;
  const int height = resized.rows;
  const int width = resized.cols;

  lines.reserve(height);

  for (int i = 0; i < height; ++i) {
    std::string line;
    line.reserve(width * 20);

    for (int j = 0; j < width; ++j) {
      const cv::Vec3b pixel = resized.at<cv::Vec3b>(i, j);
      std::ostringstream oss;
      oss << "\x1b[48;2;" << static_cast<int>(pixel[2]) << ';'
          << static_cast<int>(pixel[1]) << ';' << static_cast<int>(pixel[0])
          << "m \x1b[0m";
      line += oss.str();
    }
    lines.emplace_back(std::move(line));
  }
  return lines;
}

std::vector<std::string> Sakura::renderAsciiGrayscale(const cv::Mat &resized,
                                                      std::string_view charSet,
                                                      DitherMode dither) const {
  std::vector<std::string> lines;
  cv::Mat gray;

  if (resized.channels() == 3) {
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = resized;
  }

  const int height = gray.rows;
  const int width = gray.cols;
  const int num_chars = static_cast<int>(charSet.size());

  lines.reserve(height);

  if (dither == FLOYD_STEINBERG) {
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    cv::Mat error = cv::Mat::zeros(height, width, CV_32F);

    for (int i = 0; i < height; ++i) {
      std::string line;
      line.reserve(width);

      for (int j = 0; j < width; ++j) {
        // TODO: read more:
        // https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
        float old_value = std::clamp(
            gray.at<float>(i, j) + error.at<float>(i, j), 0.0f, 1.0f);
        int level = std::clamp(
            static_cast<int>(std::round(old_value * (num_chars - 1))), 0,
            num_chars - 1);
        const float chosen_value = static_cast<float>(level) / (num_chars - 1);
        const float err = old_value - chosen_value;

        // Distribute error
        if (j + 1 < width) {
          error.at<float>(i, j + 1) += err * 7.0f / 16.0f;
        }
        if (i + 1 < height) {
          if (j - 1 >= 0) {
            error.at<float>(i + 1, j - 1) += err * 3.0f / 16.0f;
          }
          error.at<float>(i + 1, j) += err * 5.0f / 16.0f;
          if (j + 1 < width) {
            error.at<float>(i + 1, j + 1) += err * 1.0f / 16.0f;
          }
        }
        line += charSet[level];
      }
      lines.emplace_back(std::move(line));
    }
  } else {
    for (int i = 0; i < height; ++i) {
      std::string line;
      line.reserve(width);

      for (int j = 0; j < width; ++j) {
        const int intensity = gray.at<uchar>(i, j);
        const int idx = (intensity * (num_chars - 1)) / 255;
        line += charSet[idx];
      }
      lines.emplace_back(std::move(line));
    }
  }
  return lines;
}

std::vector<std::string>
Sakura::renderImageToLines(const cv::Mat &img,
                           const RenderOptions &options) const {
  cv::Mat resized;
  int target_width, target_height;
  if (!preprocessAndResize(img, options, resized, target_width,
                           target_height)) {
    std::cerr << "Resize failed" << std::endl;
    return {};
  }

  if ((options.mode == EXACT || options.mode == ASCII_COLOR) &&
      resized.channels() == 1) {
    cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
  }

  switch (options.mode) {
  case EXACT:
    return renderExact(resized, target_height);
  case ASCII_COLOR:
    return renderAsciiColor(resized);
  case ASCII_GRAY: {
    const std::string &charSet = getCharSet(options.style);
    return renderAsciiGrayscale(resized, charSet, options.dither);
  }
  default:
    return {};
  }
}

cv::Mat Sakura::quantizeImage(const cv::Mat &inputImg, int numColors,
                              cv::Mat &palette) const {
  // this part is just textbook implementation for noow
  cv::Mat sourceImg;
  if (inputImg.channels() == 1) {
    cv::cvtColor(inputImg, sourceImg, cv::COLOR_GRAY2BGR);
  } else {
    sourceImg = inputImg;
  }

  cv::Mat workingImg;
  constexpr int MAX_PIXELS = 65536;
  if (sourceImg.rows * sourceImg.cols > MAX_PIXELS) {
    const double scale = std::sqrt(static_cast<double>(MAX_PIXELS) /
                                   (sourceImg.rows * sourceImg.cols));
    const int newWidth = std::max(1, static_cast<int>(sourceImg.cols * scale));
    const int newHeight = std::max(1, static_cast<int>(sourceImg.rows * scale));
    cv::resize(sourceImg, workingImg, cv::Size(newWidth, newHeight), 0, 0,
               cv::INTER_AREA);
  } else {
    workingImg = sourceImg;
  }

  cv::Mat samples = workingImg.reshape(1, workingImg.rows * workingImg.cols);
  samples.convertTo(samples, CV_32F);

  cv::Mat labels, centers;
  cv::kmeans(samples, numColors, labels,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              20, 1.0),
             5, cv::KMEANS_PP_CENTERS, centers);

  centers.convertTo(palette, CV_8UC1);
  palette = palette.reshape(3, numColors);

  cv::Mat quantizedImg(sourceImg.size(), CV_8U);
  for (int y = 0; y < sourceImg.rows; ++y) {
    for (int x = 0; x < sourceImg.cols; ++x) {
      const cv::Vec3b pixel = sourceImg.at<cv::Vec3b>(y, x);

      int bestIdx = 0;
      double minDistSq = std::numeric_limits<double>::max();

      for (int i = 0; i < numColors; ++i) {
        const cv::Vec3b paletteColor = palette.at<cv::Vec3b>(i);
        double distSq = 0;
        for (int c = 0; c < 3; ++c) {
          const double diff = pixel[c] - paletteColor[c];
          distSq += diff * diff;
        }
        if (distSq < minDistSq) {
          minDistSq = distSq;
          bestIdx = i;
        }
      }
      quantizedImg.at<uchar>(y, x) = static_cast<uchar>(bestIdx);
    }
  }

  return quantizedImg;
}

namespace {
int string_writer(char *data, int size, void *priv) {
  auto *sixel_string = static_cast<std::string *>(priv);
  sixel_string->append(data, size);
  return size;
}
} // namespace

std::string Sakura::renderSixel(const cv::Mat &img, int paletteSize) const {
  if (img.empty()) {
    return "";
  }

  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

  std::string sixel_output_string;
  sixel_output_string.reserve(1024 * 1024); // Pre-allocate :3

  // Use RAII for sixel resources
  struct SixelOutputDeleter {
    void operator()(sixel_output_t *p) const {
      if (p)
        sixel_output_unref(p);
    }
  };
  struct SixelDitherDeleter {
    void operator()(sixel_dither_t *p) const {
      if (p)
        sixel_dither_unref(p);
    }
  };

  std::unique_ptr<sixel_output_t, SixelOutputDeleter> output;
  std::unique_ptr<sixel_dither_t, SixelDitherDeleter> dither;

  {
    sixel_output_t *raw_output = nullptr;
    if (sixel_output_new(&raw_output, string_writer, &sixel_output_string,
                         nullptr) != SIXEL_OK) {
      return "";
    }
    output.reset(raw_output);
  }

  {
    sixel_dither_t *raw_dither = nullptr;
    if (sixel_dither_new(&raw_dither, paletteSize, nullptr) != SIXEL_OK ||
        raw_dither == nullptr) {
      return "";
    }
    dither.reset(raw_dither);
  }

  if (sixel_dither_initialize(dither.get(), rgb_img.data, rgb_img.cols,
                              rgb_img.rows, SIXEL_PIXELFORMAT_RGB888,
                              SIXEL_LARGE_NORM, SIXEL_REP_CENTER_BOX,
                              SIXEL_QUALITY_HIGH) != SIXEL_OK) {
    return "";
  }

  if (sixel_encode(rgb_img.data, rgb_img.cols, rgb_img.rows, 3, dither.get(),
                   output.get()) != SIXEL_OK) {
    return "";
  }

  return sixel_output_string;
}

bool Sakura::renderGridFromUrls(const std::vector<std::string> &urls, int cols,
                                const RenderOptions &options) const {
  if (urls.empty() || cols <= 0) {
    std::cerr << "Invalid grid parameters" << std::endl;
    return false;
  }

  const int rows = (static_cast<int>(urls.size()) + cols - 1) / cols;
  const auto [term_width, term_height] = getTerminalSize();
  const int cell_width = term_width / cols;
  const int cell_height = term_height / rows;

  std::vector<std::vector<std::string>> all_lines;
  all_lines.reserve(urls.size());

  for (const auto &url : urls) {
    const auto response = cpr::Get(cpr::Url{url});
    if (response.status_code != 200) {
      std::cerr << "Failed to download image: " << url << std::endl;
      continue;
    }

    const std::vector<uchar> imgData(response.text.begin(),
                                     response.text.end());
    const cv::Mat img = cv::imdecode(imgData, cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cerr << "Failed to decode image: " << url << std::endl;
      continue;
    }

    RenderOptions cell_options = options;
    cell_options.width = cell_width;
    cell_options.height = cell_height;
    all_lines.emplace_back(renderImageToLines(img, cell_options));
  }

  const int max_height = std::max_element(all_lines.begin(), all_lines.end(),
                                          [](const auto &a, const auto &b) {
                                            return a.size() < b.size();
                                          })
                             ->size();

  for (int i = 0; i < max_height; ++i) {
    std::string row_line;
    row_line.reserve(term_width);

    for (int c = 0; c < cols && c < static_cast<int>(all_lines.size()); ++c) {
      if (i < static_cast<int>(all_lines[c].size())) {
        row_line += all_lines[c][i];
      } else {
        row_line += std::string(cell_width, ' ');
      }
    }
    std::cout << row_line << '\n';
  }
  std::cout.flush();
  return true;
}

bool Sakura::renderGifFromUrl(std::string_view gifUrl,
                              const RenderOptions &options) const {
  cv::VideoCapture cap{std::string(gifUrl)};
  if (!cap.isOpened()) {
    std::cerr << "Failed to open GIF" << std::endl;
    return false;
  }

  const int gif_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int gif_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0)
    fps = 10.0; // Default GIF speed

  const double gifAspect = static_cast<double>(gif_width) / gif_height;
  const double termAspect = static_cast<double>(options.width) / options.height;

  RenderOptions gifOptions = options;

  if (gifAspect > termAspect) {
    gifOptions.width = options.width;
    gifOptions.height = static_cast<int>(options.width / gifAspect);
  } else {
    gifOptions.height = options.height;
    gifOptions.width = static_cast<int>(options.height * gifAspect);
  }

  if (fps > 20.0) {
    gifOptions.width = static_cast<int>(gifOptions.width * 0.95);
    gifOptions.height = static_cast<int>(gifOptions.height * 0.95);
  }

  const auto frame_duration_ns =
      std::chrono::nanoseconds(static_cast<long long>(1000000000.0 / fps));
  const auto start_time = std::chrono::high_resolution_clock::now();

  int frame_number = 0;
  int frames_dropped = 0;

  std::cout.setf(std::ios::unitbuf); // Unbuffered output
  std::cout << "\033[2J\033[?25l" << std::flush;

  cv::Mat frame, resized_frame;
  const cv::Size target_size(gifOptions.width, gifOptions.height);

  while (cap.read(frame)) {
    // time syncing
    const auto frame_start = std::chrono::high_resolution_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(frame_start -
                                                             start_time);

    const long long target_frame =
        elapsed_ns.count() / frame_duration_ns.count();

    if (frame_number < target_frame) {
      const int frames_behind = static_cast<int>(target_frame - frame_number);
      if (frames_behind > 2 && frames_dropped < frame_number * 0.3) {
        frame_number++;
        frames_dropped++;
        continue;
      }
    }

    cv::resize(frame, resized_frame, target_size, 0, 0, cv::INTER_NEAREST);

    const std::string sixel_data =
        renderSixel(resized_frame, gifOptions.paletteSize);
    std::cout << "\033[H" << sixel_data;

    frame_number++;

    const auto next_frame_time = start_time + frame_number * frame_duration_ns;
    const auto now = std::chrono::high_resolution_clock::now();

    if (next_frame_time > now) {
      const auto sleep_duration = next_frame_time - now;
      if (sleep_duration > std::chrono::milliseconds(2)) {
        std::this_thread::sleep_for(sleep_duration -
                                    std::chrono::milliseconds(1));
        while (std::chrono::high_resolution_clock::now() < next_frame_time) {
          std::this_thread::yield();
        }
      } else {
        while (std::chrono::high_resolution_clock::now() < next_frame_time) {
          std::this_thread::yield();
        }
      }
    }
  }

  std::cout << "\033[?25h" << std::flush;
  std::cout.unsetf(std::ios::unitbuf);

  cap.release();
  return true;
}

bool Sakura::renderVideoFromUrl(std::string_view videoUrl,
                                const RenderOptions &options,
                                std::string headers) const {
  std::pair<std::string, std::string> hv = splitHeader(headers);
  std::string header = hv.first;
  std::string value  = hv.second;
  const auto response = cpr::Get(cpr::Url{std::string(videoUrl)}, cpr::Header{{header, value}});
  if (response.status_code != 200) {
    std::cerr << "Failed to download video. Status: " << response.status_code
              << std::endl;
    return false;
  }

  const std::string tempFile =
      "/tmp/sakura_video_" + std::to_string(std::time(nullptr));
  std::ofstream file(tempFile, std::ios::binary);
  file.write(response.text.data(), response.text.size());
  file.close();

  const bool result = renderVideoFromFile(tempFile, options);

  std::remove(tempFile.c_str());
  return result;
}

bool Sakura::renderVideoFromFile(std::string_view videoPath,
                                 const RenderOptions &options) const {
  cv::VideoCapture cap{std::string(videoPath)};
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video: " << videoPath << std::endl;
    return false;
  }

  // video properties
  double fps = cap.get(cv::CAP_PROP_FPS);
  const int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  const int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double duration = frame_count / fps;

  if (fps <= 0)
    fps = 30.0; // Default fallback
  RenderOptions videoOptions = options;

  double scale_factor = 0.95;
  if (fps > 30.0)
    scale_factor = 0.90;
  if (fps > 50.0)
    scale_factor = 0.85;

  videoOptions.width = static_cast<int>(options.width * scale_factor);
  videoOptions.height = static_cast<int>(options.height * scale_factor);

  const double videoAspect = static_cast<double>(video_width) / video_height;
  if (videoAspect > 1.0) {
    videoOptions.height = static_cast<int>(videoOptions.width / videoAspect);
  } else {
    videoOptions.width = static_cast<int>(videoOptions.height * videoAspect);
  }

  cv::Mat frame, resized_frame;
  const cv::Size target_size(videoOptions.width, videoOptions.height);

  const auto frame_duration_ns =
      std::chrono::nanoseconds(static_cast<long long>(1000000000.0 / fps));
  const auto start_time = std::chrono::high_resolution_clock::now();

  int frame_number = 0;
  int frames_dropped = 0;
  int frames_rendered = 0;

  std::cout.setf(std::ios::unitbuf);
  std::cout << "\033[2J\033[?25l" << std::flush;

  std::atomic<bool> audio_started{false};
  auto audio_future = std::async(std::launch::async, [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    audio_started = true;
    const std::string audio_cmd = "ffplay -nodisp -autoexit \"" +
                                  std::string(videoPath) + "\" 2>/dev/null";
    std::system(audio_cmd.c_str());
  });

  std::string sixel_data;
  sixel_data.reserve(1024 * 1024);

  while (cap.read(frame)) {
    const auto frame_start = std::chrono::high_resolution_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(frame_start -
                                                             start_time);

    const long long target_frame_number =
        elapsed_ns.count() / frame_duration_ns.count();

    if (frame_number < target_frame_number) {
      const int frames_behind =
          static_cast<int>(target_frame_number - frame_number);

      if (frames_behind > 2) {
        while (frame_number < target_frame_number - 1) {
          if (!cap.read(frame))
            goto end_playback;
          frame_number++;
          frames_dropped++;
        }
      }
    }

    frame_number++;

    cv::resize(frame, resized_frame, target_size, 0, 0, cv::INTER_NEAREST);

    sixel_data = renderSixel(resized_frame, videoOptions.paletteSize);

    std::cout << "\033[H" << sixel_data;
    frames_rendered++;

    const auto next_frame_time = start_time + frame_number * frame_duration_ns;
    const auto now = std::chrono::high_resolution_clock::now();

    if (next_frame_time > now) {
      const auto sleep_duration = next_frame_time - now;
      if (sleep_duration > std::chrono::milliseconds(1)) {
        std::this_thread::sleep_for(sleep_duration);
      }
    }
  }

end_playback:

  std::cout << "\033[?25h" << std::flush;
  std::cout.unsetf(std::ios::unitbuf);
  cap.release();
  std::system("pkill -f ffplay 2>/dev/null");

  const double drop_rate = (frames_dropped * 100.0) / frame_number;
  const double render_rate = (frames_rendered * 100.0) / frame_number;
  std::cout << "\nPlayback completed. Total: " << frame_number
            << ", Rendered: " << frames_rendered << " (" << std::fixed
            << std::setprecision(1) << render_rate << "%)"
            << ", Dropped: " << frames_dropped << " (" << drop_rate << "%)"
            << std::endl;
  return true;
}
