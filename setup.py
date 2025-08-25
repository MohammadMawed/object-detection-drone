#!/usr/bin/env python3
"""
Setup configuration for Pi Drone Vision System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="pi-drone-vision",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Raspberry Pi computer vision system for drones with face recognition and object tracking",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pi-drone-vision",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/pi-drone-vision/issues",
        "Documentation": "https://github.com/yourusername/pi-drone-vision/wiki",
        "Source Code": "https://github.com/yourusername/pi-drone-vision",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Environment :: Console",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "performance": [
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pi-drone-vision=drone_vision:main",
        ],
    },
    keywords=[
        "raspberry-pi",
        "computer-vision",
        "face-recognition", 
        "object-tracking",
        "drone",
        "opencv",
        "picamera",
        "surveillance",
        "real-time",
    ],
    include_package_data=True,
    zip_safe=False,
    platforms=["Linux"],
)
