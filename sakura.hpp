#ifndef SAKURA_HPP
#define SAKURA_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

class Sakura {
public:
  enum CharStyle { SIMPLE, DETAILED, BLOCKS };
  enum RenderMode { EXACT, ASCII_COLOR, ASCII_GRAY, SIXEL };
  enum DitherMode { NONE, FLOYD_STEINBERG };

  struct RenderOptions {
    int width = 0;
    int height = 0;
    int paletteSize = 256; // For SIXEL: number of colors in palette
    CharStyle style = SIMPLE;
    RenderMode mode = EXACT;
    DitherMode dither = NONE;
    bool aspectRatio = true;
    double contrast = 1.2;
    double brightness = 0.0;
    double terminalAspectRatio = 1.0;
  };

  static std::pair<std::string, std::string> splitHeader(const std::string &headerLine);
  bool renderFromUrl(std::string_view url, const RenderOptions &options) const;
  bool renderFromUrl(std::string_view url) const;
  bool renderFromMat(const cv::Mat &img, const RenderOptions &options) const;
  bool renderGridFromUrls(const std::vector<std::string> &urls, int cols,
                          const RenderOptions &options) const;
  bool renderGifFromUrl(std::string_view gifUrl,
                        const RenderOptions &options) const;
  bool renderVideoFromUrl(std::string_view videoUrl,
                          const RenderOptions &options,
                          std::string headers) const;
  bool renderVideoFromFile(std::string_view videoPath,
                           const RenderOptions &options) const;
  std::vector<std::string> renderImageToLines(const cv::Mat &img,
                                              const RenderOptions &options) const;

private:
  static const std::string ASCII_CHARS_SIMPLE;
  static const std::string ASCII_CHARS_DETAILED;
  static const std::string ASCII_CHARS_BLOCKS;

  const std::string &getCharSet(CharStyle style) const noexcept;
  static std::pair<int, int> getTerminalSize();
  std::vector<std::string> renderExact(const cv::Mat &resized,
                                       int terminal_height) const;
  std::vector<std::string> renderAsciiColor(const cv::Mat &resized) const;
  std::vector<std::string> renderAsciiGrayscale(const cv::Mat &resized,
                                                std::string_view charSet,
                                                DitherMode dither) const;
  std::string renderSixel(const cv::Mat &img, int paletteSize = 16) const;
  cv::Mat quantizeImage(const cv::Mat &inputImg, int numColors,
                        cv::Mat &palette) const;
  bool preprocessAndResize(const cv::Mat &img, const RenderOptions &options,
                           cv::Mat &resized, int &target_width,
                           int &target_height) const;
};

#endif // SAKURA_HPP
