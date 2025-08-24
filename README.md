# Sakura 

A high-performance minimal terminal-based multimedia library that renders images, GIFs, and videos with **SIXEL graphics** and **enhanced ASCII rendering** modes. Features real-time audio playback synchronization and advanced rendering options.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Implementation](#technical-implementation)
- [Performance Optimizations](#performance-optimizations)
- [SIXEL Terminal Support](#sixel-terminal-support)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)


Video showcasing rendering video with smooth playback:
<video src='examples/video.mp4' width=180/>
## Features

### Core Capabilities

- **SIXEL Graphics Rendering**: Pixel-perfect graphics directly in the terminal using libsixel
- **Multi-format Support**: Images (JPG, PNG, BMP), animated GIFs, and videos (MP4, AVI, MOV, MKV)
- **Synchronized Audio**: Real-time audio playbook with video using ffmpeg
- **URL Download**: Direct streaming from web URLs
- **Multiple Rendering Modes**: SIXEL, enhanced ASCII, ASCII color, and grayscale modes
- **Adaptive Scaling**: Terminal-aware sizing with aspect ratio preservation
- **Performance Optimizations**: Predecode queue, frame pacing, and adaptive palette

### Rendering Modes

1. **EXACT Mode**: Enhanced ASCII rendering with block characters and precise terminal fitting
2. **SIXEL Mode**: Pixel-perfect graphics with full color palette for supported terminals  
3. **ASCII_COLOR Mode**: Block-based color rendering with 24-bit RGB
4. **ASCII_GRAY Mode**: Character-based monochrome with dithering support

### Character Sets

- **SIMPLE**: Basic ASCII characters (` .:-=+*#%@`)
- **DETAILED**: Extended ASCII set with 69 characters
- **BLOCKS**: Unicode block characters (`░▒▓█`)
- **ULTRA**: High-quality blocks (`▁▂▃▄▅▆▇█`)
- **MICRO**: Ultra-fine detail characters for premium quality

### Advanced Features

- **Smart Dithering**: Floyd-Steinberg and Atkinson dithering algorithms
- **Fit Modes**: STRETCH, COVER, CONTAIN for optimal terminal usage
- **Hardware Acceleration**: Optional ffmpeg hardware decode pipeline
- **Tiled Updates**: Send only changed regions for better performance
- **Adaptive Quality**: Dynamic palette and scale adjustments

## Installation

### Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libopencv-dev libsixel-dev ffmpeg cmake build-essential

# Arch Linux
sudo pacman -S opencv sixel cpr ffmpeg cmake base-devel

# macOS (Homebrew)
brew install opencv libsixel cpr ffmpeg cmake
```

> [!NOTE]
> For Ubuntu/Debian users: `cpr` is not available as a package in Ubuntu/Debian repositories.  
You must clone and build it manually.

```bash
git clone https://github.com/libcpr/cpr.git
cd cpr
mkdir build && cd build
cmake ..
make
sudo make install
```

Alternatively, clone `cpr` into your project root and include it as a subdirectory in your `CMakeLists.txt`

#### Build Instructions

```bash
git clone https://github.com/Sarthak2143/sakura.git
cd sakura
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Quick Start

```bash
./sakura --help
./sakura --image https://picsum.photos/800/600
```

### NixOS

#### Flakes (recommended)
install module via flakes
```nix
{
  inputs.sakura.url = "github:sarthak2143/sakura";
  inputs.sakura.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { self, nixpkgs, sakura }: {
    # change `yourhostname` to your actual hostname
    nixosConfigurations.yourhostname = nixpkgs.lib.nixosSystem {
      # customize to your system
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        sakura.nixosModules.sakura
      ];
    };
  };
}

# enable it in your configuration
programs.sakura.enable = true;
```

#### Home Manager 
install home-manager modules via flakes
```nix
{
  inputs.sakura.url = "github:sarthak2143/sakura";
  inputs.sakura.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { self, nixpkgs, home-manager, sakura }: {
    homeConfigurations."username" = home-manager.lib.homeManagerConfiguration {
      # ...
      modules = [
        sakura.homeModules.sakura
        # ...
      ];
    };
  };
}

# enable it in home manager config
programs.sakura.enable = true;
```

#### Install CLI via flakes
you can run sakura ad-hoc without installing it.
```bash
nix run github:sarthak2143/sakura
```

you can also install it into NixOS modules
```nix
{
  inputs.sakura.url = "github:sarthak2143/sakura";
  inputs.sakura.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { self, nixpkgs, sakura }: {
    # change `yourhostname` to your actual hostname
    nixosConfigurations.yourhostname = nixpkgs.lib.nixosSystem {
      # customize to your system
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        {
          environment.systemPackages = [ sakura.packages.${system}.default ];
        }
      ];
    };
  };
}
```

## Usage

### Command Line Interface

```bash
# Show help
./sakura --help

# Process single image
./sakura --image https://example.com/image.jpg

# Process GIF animation  
./sakura --gif https://example.com/animation.gif

# Process video from URL
./sakura --video https://example.com/video.mp4

# Process local video file
./sakura --local-video /path/to/video.mp4
```

### Interactive Menu

When you run `./sakura` without arguments, you'll see an interactive menu:

```text
Sakura Video Player with SIXEL
1. Image
2. GIF  
3. Video (URL)
4. Video (File)
Choose option (1-4):
```

### Usage Examples

## Technical Implementation

### Architecture

Sakura uses a modern C++ architecture with the following key components:

- **OpenCV**: Image/video processing and codec handling
- **libsixel**: High-quality SIXEL graphics encoding
- **cpr**: HTTP client for URL downloads
- **ffmpeg**: Audio playback synchronization

### Rendering Pipeline

1. **Input Processing**: Image loading, URL downloading, or video frame extraction
2. **Preprocessing**: Contrast/brightness adjustment, aspect ratio calculation
3. **Scaling**: Terminal-aware resizing with configurable interpolation
4. **Mode Selection**: SIXEL, EXACT, ASCII_COLOR, or ASCII_GRAY rendering
5. **Character Mapping**: Smart luminance-to-character conversion (ASCII modes)
6. **Dithering**: Optional Floyd-Steinberg or Atkinson error diffusion
7. **Output**: ANSI escape sequences or SIXEL data streams

### Performance Optimizations

- **Predecode Queue**: Background threading for video frame preprocessing
- **Adaptive Quality**: Dynamic palette and scale adjustments under load  
- **Frame Pacing**: Precise timing control with `std::chrono::steady_clock`
- **Memory Management**: Pre-allocated buffers and efficient string handling
- **Hardware Acceleration**: Optional ffmpeg hardware decode pipeline
- **Tiled Updates**: Differential rendering for animated content

## SIXEL Terminal Support

### Compatible Terminals

| Terminal | SIXEL Support | Command |
|----------|---------------|---------|
| **xterm** | Native | `xterm -ti vt340` |
| **mlterm** | Native | Default |
| **wezterm** | Configurable | Enable in config |
| **foot** | Native | Default |
| **mintty** | Optional | `--enable-sixel` |
| **iTerm2** | Beta | Enable in preferences |

### Terminal Configuration

#### xterm Setup
```bash
# Launch with SIXEL support
xterm -ti vt340 -geometry 120x40

# Or add to ~/.Xresources
xterm*decTerminalID: vt340
```

#### wezterm Configuration
```lua
-- ~/.config/wezterm/wezterm.lua
return {
  enable_sixel = true,
  max_fps = 60,
}
```

## API Documentation

### Core Classes

#### `Sakura` Class

Main rendering engine with the following public methods:

```cpp
class Sakura {
public:
    // Image rendering
    bool renderFromUrl(std::string_view url, const RenderOptions &options) const;
    bool renderFromUrl(std::string_view url) const;  // Uses default options
    bool renderFromMat(const cv::Mat &img, const RenderOptions &options) const;
    
    // Grid rendering
    bool renderGridFromUrls(const std::vector<std::string> &urls, int cols,
                           const RenderOptions &options) const;
    
    // Video/GIF rendering  
    bool renderGifFromUrl(std::string_view gifUrl, const RenderOptions &options) const;
    bool renderVideoFromUrl(std::string_view videoUrl, const RenderOptions &options) const;
    bool renderVideoFromFile(std::string_view videoPath, const RenderOptions &options) const;
    
    // Utility methods
    std::vector<std::string> renderImageToLines(const cv::Mat &img, 
                                               const RenderOptions &options) const;
};
```

#### `RenderOptions` Structure

```cpp
struct RenderOptions {
    int width = 0;                     // Target width (0 = auto)
    int height = 0;                    // Target height (0 = auto)
    int paletteSize = 256;             // SIXEL palette size
    CharStyle style = SIMPLE;          // Character set style
    RenderMode mode = EXACT;           // Rendering mode
    DitherMode dither = NONE;          // Dithering algorithm
    bool aspectRatio = true;           // Preserve aspect ratio
    double contrast = 1.2;             // Contrast adjustment
    double brightness = 0.0;           // Brightness adjustment
    double terminalAspectRatio = 1.0;  // Terminal character aspect
    int queueSize = 16;                // Predecode queue size
    int prebufferFrames = 4;           // Frames to prebuffer
    bool staticPalette = false;        // Reuse first palette
    FitMode fit = COVER;               // Scaling behavior
    bool fastResize = false;           // Use INTER_NEAREST
    SixelQuality sixelQuality = HIGH;  // SIXEL quality setting
    
    // Performance controls
    double targetFps = 0.0;            // Target FPS (0 = source FPS)
    bool adaptivePalette = false;      // Dynamic palette sizing
    int minPaletteSize = 64;           // Minimum palette size
    int maxPaletteSize = 256;          // Maximum palette size
    bool adaptiveScale = false;        // Dynamic scaling
    double minScaleFactor = 0.80;      // Minimum scale factor
    double maxScaleFactor = 1.00;      // Maximum scale factor
    double scaleStep = 0.05;           // Scale adjustment step
    
    // Advanced features
    bool hwAccelPipe = false;          // Hardware acceleration
    bool tileUpdates = false;          // Tiled rendering
    int tileWidth = 128;               // Tile width
    int tileHeight = 64;               // Tile height
    double tileDiffThreshold = 6.0;    // Tile change threshold
};
```

### Enums

```cpp
enum CharStyle { SIMPLE, DETAILED, BLOCKS, ULTRA, MICRO };
enum RenderMode { EXACT, ASCII_COLOR, ASCII_GRAY, SIXEL };
enum DitherMode { NONE, FLOYD_STEINBERG, ATKINSON };
enum FitMode { STRETCH, COVER, CONTAIN };
enum SixelQuality { LOW, HIGH };
```

## Examples

### Programmatic Usage

```cpp
#include "sakura.hpp"

int main() {
    Sakura renderer;
    Sakura::RenderOptions options;
    
    // Configure for high-quality SIXEL rendering
    options.mode = Sakura::SIXEL;
    options.sixelQuality = Sakura::HIGH;
    options.paletteSize = 256;
    options.dither = Sakura::FLOYD_STEINBERG;
    options.fit = Sakura::COVER;
    options.aspectRatio = true;
    
    // Render image from URL
    renderer.renderFromUrl("https://example.com/image.jpg", options);
    
    // Render local video file
    renderer.renderVideoFromFile("video.mp4", options);
    
    return 0;
}
```

### Custom ASCII Rendering

```cpp
// Configure for enhanced ASCII mode
Sakura::RenderOptions options;
options.mode = Sakura::EXACT;           // Enhanced ASCII
options.style = Sakura::ULTRA;          // Ultra-quality blocks
options.dither = Sakura::ATKINSON;      // Advanced dithering
options.contrast = 1.2;
options.brightness = 10;

renderer.renderFromMat(image, options);
```

### Performance-Optimized Video

```cpp
// High-performance video settings
Sakura::RenderOptions options;
options.mode = Sakura::SIXEL;
options.fastResize = true;              // Fast scaling
options.targetFps = 30.0;               // Stable framerate
options.queueSize = 48;                 // Large buffer
options.adaptivePalette = true;         // Dynamic quality
options.hwAccelPipe = true;             // Hardware decode
options.tileUpdates = true;             // Efficient updates

renderer.renderVideoFromFile("large_video.mp4", options);
```

## TODO

- [ ] Add error handling and exception classes
- [ ] Implement unit test suite
- [ ] Add configuration file support
- [ ] GPU acceleration for image processing
- [ ] WebM and additional codec support
- [ ] Real-time streaming input support
- [ ] Plugin architecture for custom renderers
- [ ] Terminal capability auto-detection
- [ ] Improved memory management for large videos
- [ ] Cross-platform Windows support optimization

## Troubleshooting

### Common Issues

#### "Failed to open video"
```bash
# Check file exists and permissions
ls -la /path/to/video.mp4

# Verify OpenCV codec support
ffmpeg -codecs | grep h264

# Try different format
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

#### No SIXEL output
```bash
# Test terminal SIXEL support
echo -e '\ePq"1;1;100;100#0;2;0;0;0#1;2;100;100;0#1~~@@vv@@~~@@~~$#0~~@@~~@@~~@@vv$#1!14~\e\\'

# Launch with SIXEL-capable terminal
xterm -ti vt340 -e ./sakura
```

#### Audio/video sync issues
```bash
# Check ffplay installation
which ffplay

# Test audio playback separately
ffplay -nodisp -autoexit video.mp4

# Check audio permissions (containers)
pulseaudio --check
```

#### Performance issues
```bash
# Reduce video resolution
ffmpeg -i input.mp4 -vf scale=640:480 output.mp4

# Lower frame rate
ffmpeg -i input.mp4 -r 15 output.mp4

# Use hardware acceleration if available
ffmpeg -hwaccel auto -i input.mp4 output.mp4
```

### Debug Mode

Enable debug builds for development:

```bash
# Build with debug information
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Run with debugging tools
gdb ./sakura
valgrind --leak-check=full ./sakura
```

## Contributing

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Sarthak2143/sakura.git
cd sakura

# Install development dependencies
sudo apt install cmake build-essential libopencv-dev libsixel-dev

# Build with debug information
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

### Testing

```bash
# Test basic functionality
./sakura --help
./sakura --image https://picsum.photos/800/600

# Test with local media files
./sakura --local-video media/example.mp4

# Memory debugging
valgrind --leak-check=full ./sakura --image test.jpg
```

## Acknowledgments

- **libsixel** - SIXEL graphics encoding library
- **OpenCV** - Computer vision and image processing
- **FFmpeg** - Multimedia framework for audio/video
- **cpr** - C++ HTTP request library

---
