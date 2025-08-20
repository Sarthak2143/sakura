#ifndef SAKURA_HPP
#define SAKURA_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>
#include <vector>

class Sakura {
public:
  enum CharStyle { SIMPLE, DETAILED, BLOCKS };
  enum RenderMode { EXACT, ASCII_COLOR, ASCII_GRAY, SIXEL };
  enum DitherMode { NONE, FLOYD_STEINBERG };
  enum FitMode { STRETCH, COVER, CONTAIN };

  enum SixelQuality { LOW, HIGH };

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
    int queueSize = 16;
    int prebufferFrames = 4;
    bool staticPalette = false;
    FitMode fit = COVER;
    bool fastResize = false; // Use INTER_NEAREST for video pre-scaling
    SixelQuality sixelQuality = HIGH; // Add this line
    // Throughput controls
    double targetFps =
        0.0; // 0 = follow source FPS; otherwise downsample to this
    bool adaptivePalette = false;
    int minPaletteSize = 64;      // when adaptivePalette is true
    int maxPaletteSize = 256;     // cap
    bool adaptiveScale = false;   // dynamically adjust scale to keep up
    double minScaleFactor = 0.80; // 80% of computed size
    double maxScaleFactor = 1.00; // up to full size
    double scaleStep = 0.05;      // adjust step per window
    // Hardware-accelerated decode and tiled updates
    bool hwAccelPipe = false;     // use ffmpeg pipe with -hwaccel auto
    bool tileUpdates = false;     // send only changed tiles each frame
    int tileWidth = 128;          // tile width in pixels
    int tileHeight = 64;          // tile height in pixels
    double tileDiffThreshold = 6.0; // average abs diff per channel to trigger update
  };

  bool renderFromUrl(std::string_view url, const RenderOptions &options) const;
  bool renderFromUrl(std::string_view url) const;
  bool renderFromMat(const cv::Mat &img, const RenderOptions &options) const;
  bool renderGridFromUrls(const std::vector<std::string> &urls, int cols,
                          const RenderOptions &options) const;
  bool renderGifFromUrl(std::string_view gifUrl,
                        const RenderOptions &options) const;
  bool renderVideoFromUrl(std::string_view videoUrl,
                          const RenderOptions &options) const;
  bool renderVideoFromFile(std::string_view videoPath,
                           const RenderOptions &options) const;
  std::vector<std::string>
  renderImageToLines(const cv::Mat &img, const RenderOptions &options) const;

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
  std::string renderSixel(const cv::Mat &img, int paletteSize = 16, int output_width = 0, int output_height = 0, SixelQuality quality = HIGH) const;
  std::string renderVideoEnhanced(const cv::Mat &frame, const RenderOptions &options) const; // Enhanced video method
  std::string renderExactVideo(const cv::Mat &frame) const; // Helper for exact video rendering
  std::string renderAsciiColorVideo(const cv::Mat &frame) const; // Helper for color ASCII video
  std::string renderAsciiGrayVideo(const cv::Mat &frame, const RenderOptions &options) const; // Helper for grayscale ASCII video
  cv::Mat quantizeImage(const cv::Mat &inputImg, int numColors,
                        cv::Mat &palette) const;
  bool preprocessAndResize(const cv::Mat &img, const RenderOptions &options,
                           cv::Mat &resized, int &target_width,
                           int &target_height) const;
};

#endif // SAKURA_HPP
