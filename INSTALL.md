# Installation Process

Installation is a multipart process:
1. O/S Dependencies
    - cmake
    - gcc
    - git
    - wget (or curl) for downloading
    - g++ (on ubuntu and fedora; not required for archlinux)
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

### Install to be accessible for all users
(E.g. inside /usr/local/lib on Linux systems)

```python3 -m pip install "pybind11[global] meson[global] ninja[global]"```

### Install to be accessible for the login user only
(E.g. inside $HOME/.local/lib on Linux systems)

```python3 -m pip install -U --user pybind11 meson ninja```

## Other Dependencies

### benchmark

* Follow instructions at https://github.com/google/benchmark#installation.
* Be sure to install the library globally.

### fmt

* Download fmt from https://fmt.dev/latest/index.html.
* Currently: https://github.com/fmtlib/fmt/releases/download/9.0.0/fmt-9.0.0.zip.
* Extract zip or tar.gz, creating directory **fmt**.
* Then,
```
cd fmt
cmake `pwd`
make
sudo make install
```

## CyberEther

```
git clone https://github.com/luigifcruz/CyberEther.git
cd CyberEther
git checkout 35693631c3110dfab0327599f15922fdf0028005
meson build -Dbuildtype=release
cd build
ninja
./cyberether # Verify!
```
