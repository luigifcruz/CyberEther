# FFT Axis Selection Feature

## Overview

This document describes the implementation of the axis selection feature for the Fast Fourier Transform (FFT) module and block in CyberEther. This feature allows users to specify which axis of a multi-dimensional tensor to perform the FFT operation on, rather than always using the last axis.

## Changes Made

### 1. FFT Module Changes

#### Modified Files:
- `/include/jetstream/modules/fft.hh`
- `/src/modules/fft/generic.cc`
- `/src/modules/fft/cpu/base.cc`

#### Enhancements:
- Added an `axis` parameter to the module's configuration struct
- Default value is -1, which means "use the last axis" (preserving backward compatibility)
- Added validation to ensure the requested axis is valid for the input tensor
- Updated the module's info method to display which axis is being used
- Modified FFT operation to use the specified axis

### 2. FFT Block Changes

#### Modified Files:
- `/include/jetstream/blocks/fft.hh`

#### Enhancements:
- Added the `axis` parameter to the block's configuration struct
- Updated the block's create() method to pass the axis configuration to the module
- Enhanced the UI with a dropdown menu for axis selection, showing available options based on the input tensor
- Improved the block description with comprehensive documentation of inputs, outputs, configuration options, and supported data types

## User Interface

The FFT block's interface now includes:
- A dropdown menu for selecting FFT direction (Forward/Backward)
- A new dropdown menu for selecting the axis along which to perform the FFT:
  - "Last axis" (default)
  - Specific axis numbers (0, 1, 2, etc.) based on the input tensor's rank

## Benefits

This enhancement provides several benefits:
1. **Greater flexibility**: Users can now perform FFT operations along any axis of their data
2. **Better performance**: In some cases, choosing a specific axis may be more efficient
3. **Improved workflow**: Reduces the need for tensor reshape operations to move data to the last axis
4. **Better documentation**: The updated block description provides clearer information about the block's functionality

## Implementation Details

The implementation follows these key principles:
1. **Backward compatibility**: The default behavior (-1 = last axis) is preserved
2. **Error handling**: Invalid axis selections are detected and reported
3. **UI adaptation**: The UI shows only valid axis options based on the input tensor
4. **Cross-platform compatibility**: Implementation works across all supported backends

## Future Work

Potential future enhancements for the FFT module and block:
1. Support for multiple simultaneous axes (N-dimensional FFT)
2. Performance optimizations for specific axis configurations
3. Additional FFT parameters like normalization options

## Contributor

Implemented by Chase Valentine ([@cvalentine99](https://github.com/cvalentine99))