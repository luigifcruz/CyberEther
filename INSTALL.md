# Installation Process

Installation is a multipart process:
1. O/S Dependencies
    - cmake, gcc, git, wget (or curl) for downloading
    - g++ (ubuntu and fedora; not required for archlinux)
    - python3-pip (ubuntu, fedora) | python-pip (archlinux)
    - python3-dev (Ubuntu) | python3-devel (fedora) | python3 (archlinux)
    - pybind11 (if available as an O/S package)
    - meson (if available as an O/S package)
    - ninja (if available as an O/S package)
2. pip Dependencies (if needed)
    - pybind11 (if not available as an O/S package)
    - meson (if not available as an O/S package)
    - ninja (if not available as an O/S package)
3. Other Dependencies
    - benchmark
    - fmt
4. CyberEther

## O/S Dependencies

### Linux

Use the resident package manager such as pacman (archlinux), dnf (fedora, centos), and apt (ubuntu, debian, mint, etc.).
Install all of the O/S packages.

### MacOS

*TBD*

### Windows

*TBD*

## pip Dependencies

If the resident O/S does not have a pybind11 package, do the following:
Install pybind11 such that it is accessible to all users (E.g. /usr/local on Linux systems):
python3 -m pip install "pybind11[global]" 

If the resident O/S does not have a meson package, do the following:
python3 -m pip install "meson[global]"

If the resident O/S does not have a ninja package, do the following:
python3 -m pip install "ninja[global]"

## Other Dependencies

### benchmark

Follow instructions at https://github.com/google/benchmark#installation.
Be sure to install the library globally.

### fmt

Download fmt from https://fmt.dev/latest/index.html.
Currently: https://github.com/fmtlib/fmt/releases/download/9.0.0/fmt-9.0.0.zip
Extract zip or tar.gz.

cd fmt
cmake .
make
sudo make install

## CyberEther

git clone https://github.com/luigifcruz/CyberEther.git
cd CyberEther
git checkout 35693631c3110dfab0327599f15922fdf0028005
meson build -Dbuildtype=release
cd build
ninja
./cyberether # Verify!

