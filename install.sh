#!/bin/bash
# Object Detection Drone System Installation Script
# Optimized for Raspberry Pi 4/5

echo "🚁 Object Detection Drone System Installer"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    print_warning "This script is optimized for Raspberry Pi. Continuing anyway..."
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.7+ required. Found: $python_version"
    exit 1
fi

print_status "Python version check passed: $python_version"

# Update system packages
print_step "Updating system packages..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get upgrade -y
else
    print_warning "apt-get not found. Please update your system manually."
fi

# Install system dependencies
print_step "Installing system dependencies..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y \
        python3-pip \
        python3-dev \
        python3-venv \
        cmake \
        build-essential \
        pkg-config \
        libopencv-dev \
        libatlas-base-dev \
        liblapack-dev \
        libx11-dev \
        libgtk-3-dev \
        libboost-python-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libdc1394-22-dev \
        ffmpeg \
        v4l-utils
    
    print_status "System dependencies installed successfully"
else
    print_warning "Please install the required system dependencies manually"
fi

# Check if camera is enabled
print_step "Checking camera configuration..."
if ! vcgencmd get_camera | grep -q "detected=1"; then
    print_error "Camera not detected!"
    print_warning "Please run 'sudo raspi-config' and enable the camera interface"
    print_warning "Then reboot your Pi and run this installer again"
    echo ""
    echo "Steps to enable camera:"
    echo "1. sudo raspi-config"
    echo "2. Navigate to: Interface Options > Camera"
    echo "3. Select 'Yes' to enable"
    echo "4. Reboot: sudo reboot"
    exit 1
else
    print_status "Camera detected and enabled"
fi

# Create virtual environment (recommended)
print_step "Setting up virtual environment (recommended)..."
read -p "Create virtual environment? (y/N): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    fi
    
    source venv/bin/activate
    print_status "Virtual environment activated"
    
    # Upgrade pip in virtual environment
    pip install --upgrade pip
else
    print_warning "Installing packages system-wide"
    # Upgrade pip system-wide
    python3 -m pip install --upgrade pip --user
fi

# Install Python dependencies
print_step "Installing Python dependencies..."
echo "This may take 10-15 minutes on Raspberry Pi..."

# Install dependencies one by one for better error handling
dependencies=(
    "numpy>=1.21.0"
    "Pillow>=9.0.0"
    "opencv-python==4.8.1.78"
    "picamera2>=0.3.12"
    "dlib>=19.24.0"
    "face-recognition==1.3.0"
    "psutil>=5.9.0"
)

for dep in "${dependencies[@]}"; do
    print_status "Installing $dep..."
    if [[ $create_venv =~ ^[Yy]$ ]]; then
        pip install "$dep" || {
            print_error "Failed to install $dep"
            exit 1
        }
    else
        python3 -m pip install "$dep" --user || {
            print_error "Failed to install $dep"
            exit 1
        }
    fi
done

print_status "All Python dependencies installed successfully"

# Create necessary directories
print_step "Creating project directories..."
mkdir -p captures face_crops logs
print_status "Project directories created"

# Set up GPIO memory access (for better camera performance)
print_step "Configuring GPU memory..."
current_gpu_mem=$(vcgencmd get_mem gpu | cut -d'=' -f2 | cut -d'M' -f1)
if [ "$current_gpu_mem" -lt "128" ]; then
    print_warning "GPU memory is ${current_gpu_mem}M. Recommend 128M+ for camera operations"
    echo "To increase GPU memory:"
    echo "1. sudo raspi-config"
    echo "2. Advanced Options > Memory Split"
    echo "3. Set to 128 or 256"
    echo "4. Reboot"
fi

# Performance optimization suggestions
print_step "Performance optimization check..."
cpu_temp=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
print_status "Current CPU temperature: ${cpu_temp}°C"

if (( $(echo "$cpu_temp > 70" | bc -l) )); then
    print_warning "CPU temperature is high (${cpu_temp}°C)"
    print_warning "Consider adding cooling (fan/heatsinks) for optimal performance"
fi

# Test installation
print_step "Testing installation..."
python3 -c "
import cv2
import numpy as np
import face_recognition
from picamera2 import Picamera2
print('✓ All imports successful')
" || {
    print_error "Installation test failed"
    exit 1
}

print_status "Installation test passed!"

# Final setup
print_step "Final setup..."

# Create a simple test script
cat > test_camera.py << 'EOF'
#!/usr/bin/env python3
"""Simple camera test for Pi Drone Vision System"""
from picamera2 import Picamera2
import cv2
import time

def test_camera():
    print("Testing camera...")
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (320, 240)})
        picam2.configure(config)
        picam2.start()
        
        print("Camera started successfully!")
        print("Press 'q' to quit test")
        
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Camera test failed: {e}")
    finally:
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()
        print("Camera test completed")

if __name__ == "__main__":
    test_camera()
EOF

chmod +x test_camera.py
print_status "Camera test script created"

# Success message
echo ""
echo "🎉 Installation Complete!"
echo "======================="
print_status "Object Detection Drone System is ready to use!"
echo ""
echo "Quick Start:"
echo "1. Test camera: python3 test_camera.py"
echo "2. Run main system: python3 drone_vision.py"
echo ""
echo "If you created a virtual environment, activate it first:"
echo "source venv/bin/activate"
echo ""
echo "Optimization Tips:"
echo "• Use fast SD card (Class 10, U3)"
echo "• Ensure adequate cooling"
echo "• Close unnecessary applications"
echo "• Monitor CPU temperature during use"
echo ""
print_status "Happy flying! 🚁"

# Check if reboot needed
if [ ! -f "/tmp/pi_vision_installed" ]; then
    touch /tmp/object_detection_installed
    print_warning "A reboot may be required for optimal performance"
    read -p "Reboot now? (y/N): " reboot_now
    if [[ $reboot_now =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
fi
