# 🚁 Object Detection Drone System

An advanced, Raspberry Pi-optimized computer vision system for drones featuring real-time face recognition, object tracking, and motion detection. Built specifically for drone applications with performance optimizations for Pi hardware.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4%2F5-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🎯 **Core Capabilities**
- **Real-time Face Recognition** - Automatically detects and saves faces with persistent database
- **Object Tracking** - Track moving objects with KCF lightweight tracker
- **Motion Detection** - Background subtraction for movement detection
- **Drone HUD** - Heads-up display with crosshair and system stats
- **Performance Optimized** - Specifically tuned for Raspberry Pi 4/5

### 🚀 **Drone-Specific Features**
- **320x240 capture** resolution for optimal Pi performance
- **15 FPS target** with frame rate limiting
- **Persistent face database** remembers people across flights
- **Compressed image storage** to prevent SD card overflow
- **Session logging** with flight statistics
- **Crosshair overlay** for precise positioning

### 🧠 **Smart Processing**
- **Interval-based processing** - Faces every 10 frames, motion every 3 frames
- **Memory management** - Automatic cleanup and database trimming  
- **Fallback systems** - Multiple tracker options for reliability
- **Background operations** - Non-blocking database saves

## 📋 Requirements

### Hardware
- **Raspberry Pi 4 (4GB+)** or **Raspberry Pi 5** recommended
- **Pi Camera Module** (v2/v3 or HQ Camera)
- **Fast SD Card** (Class 10, U3, or better)
- **Adequate cooling** (fan or heatsinks recommended)

### Software
- **Raspberry Pi OS** (Bullseye or newer)
- **Python 3.7+**
- **Camera interface enabled** in `raspi-config`

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MohammadMawed/object-detection-drone.git
cd object-detection-drone
```

### 2. Run Installation Script
```bash
chmod +x install.sh
./install.sh
```

**Or install manually:**
```bash
pip install -r requirements.txt
```

### 3. Enable Pi Camera
```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
sudo reboot
```

### 4. Run the System
```bash
python drone_vision.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `S` | Save compressed snapshot |
| `T` | Start tracking detected motion object |
| `R` | Reset/stop current tracking |
| `C` | Clear face database |
| `P` | Show performance information |

## 📊 Performance

### Raspberry Pi 4 (4GB)
- **12-15 FPS** sustained performance
- **~50-60% CPU** usage during operation
- **Face detection** at ~1.5 FPS (every 10 frames)
- **Memory usage** ~150-200MB

### Raspberry Pi 5
- **15+ FPS** sustained performance  
- **~40-50% CPU** usage during operation
- **Better thermal management**
- **Faster face processing**

## 📁 Project Structure

```
object-detection-drone/
├── drone_vision.py          # Main application
├── requirements.txt         # Python dependencies
├── install.sh              # Installation script
├── setup.py                # Package setup
├── README.md               # This file
├── LICENSE                 # MIT License
├── examples/               # Example configurations
│   ├── basic_config.py     # Basic setup example
│   └── advanced_config.py  # Advanced configuration
├── docs/                   # Documentation
│   ├── INSTALLATION.md     # Detailed installation guide
│   ├── CONFIGURATION.md    # Configuration options
│   ├── TROUBLESHOOTING.md  # Common issues and fixes
│   └── API.md             # API documentation
├── tests/                  # Unit tests
│   ├── test_vision.py      # Vision system tests
│   └── test_performance.py # Performance tests
└── scripts/                # Utility scripts
    ├── benchmark.py        # Performance benchmarking
    ├── camera_test.py      # Camera functionality test
    └── cleanup.py          # Database cleanup utility
```

## ⚙️ Configuration

### Basic Configuration
```python
# In drone_vision.py, modify these settings:
self.frame_width = 320              # Capture resolution width
self.frame_height = 240             # Capture resolution height
self.target_fps = 15                # Target frame rate
self.face_detection_interval = 10   # Process faces every N frames
```

### Advanced Options
See [CONFIGURATION.md](docs/CONFIGURATION.md) for detailed configuration options.

## 🚁 Drone Integration

### Camera Mounting
- Mount camera with **forward-facing orientation**
- Ensure **vibration dampening** for clear footage
- Consider **gimbal integration** for stabilization

### Flight Considerations
- System runs **independently** of flight controller
- Can log **GPS coordinates** if integrated
- **Thermal management** important during flight

## 📈 Optimization Tips

### Hardware Optimization
- Use **fast SD card** (SanDisk Extreme Pro recommended)
- Ensure **adequate cooling** (fan + heatsinks)
- **Overclock Pi safely** if thermal management allows
- Use **quality power supply** (official Pi power supply)

### Software Optimization
- Close **unnecessary services** before running
- Set **GPU memory split** to 128MB minimum
- Consider **read-only filesystem** for flight applications
- Monitor **CPU temperature** during operation

## 🧪 Testing

Run the test suite to verify installation:
```bash
python -m pytest tests/
```

Performance benchmark:
```bash
python scripts/benchmark.py
```

Camera test:
```bash
python scripts/camera_test.py
```

## 🔧 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Check camera connection
vcgencmd get_camera

# Enable camera interface
sudo raspi-config
```

**Low FPS performance:**
- Check CPU temperature: `vcgencmd measure_temp`
- Ensure adequate cooling
- Close background applications
- Consider lowering resolution

**Face recognition errors:**
- Ensure sufficient lighting
- Check if dlib compiled correctly
- Try reducing face detection interval

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions.

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed setup instructions
- [Configuration Options](docs/CONFIGURATION.md) - All configuration parameters
- [API Reference](docs/API.md) - Code documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/MohammadMawed/object-detection-drone.git
cd object-detection-drone

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** community for computer vision tools
- **face_recognition** library by Adam Geitgey
- **Picamera2** library by Raspberry Pi Foundation
- **Raspberry Pi Foundation** for amazing hardware

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MohammadMawed/object-detection-drone/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MohammadMawed/object-detection-drone/discussions)
- **Wiki**: [Project Wiki](https://github.com/MohammadMawed/object-detection-drone/wiki)

## 🔄 Changelog

### v1.0.0 (2025-01-XX)
- Initial release
- Face recognition with persistent database
- Object tracking with KCF
- Motion detection
- Pi 4/5 optimizations
- HUD overlay system
- Session logging

---

# Install dlib from system packages instead of pip
sudo apt install -y python3-dlib

# Then install other packages via pip
source venv/bin/activate
pip install opencv-python numpy Pillow picamera2 psutil

# For face-recognition, try this lighter approach
pip install face-recognition --no-deps 

**Made with ❤️ for the drone and Raspberry Pi community**

### 📊 Stats
![GitHub stars](https://img.shields.io/github/stars/MohammadMawed/object-detection-drone?style=social)
![GitHub forks](https://img.shields.io/github/forks/MohammadMawed/object-detection-drone?style=social)
![GitHub issues](https://img.shields.io/github/issues/MohammadMawed/object-detection-drone)
