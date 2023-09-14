//
//  demo_config.h
//  CyberEtherMobile
//
//  Created by Luigi Cruz on 9/5/23.
//

#ifndef demo_config_h
#define demo_config_h


static const char* DemoConfigBlob = R"(
---
protocolVersion: 1.0.0
cyberetherVersion: 1.0.0
name: Simple FM
creator: Luigi
license: MIT

engine:
  backends:
    cpu:
    metal:
  render:
    device: metal
    config:
      scale: 0.90
  viewport:
    platform: glfw
    config:
      title: CyberEther
      size: [2048, 960]

computeDevice: cpu

graph:
  soapy:
    module: soapy-view
    device: ${computeDevice}
    dataType: CF32
    config:
      deviceString: driver=remote,remote:driver=airspy,remote=11.11.5.1
      streamString: remote:mtu=4000,remote:window=102400,remote:prot=udp
      frequency: 94.5e6
      sampleRate: 2.5e6
      outputShape: [1, 2048]
      bufferMultiplier: 512

  win:
    module: window
    device: ${computeDevice}
    dataType: CF32
    config:
      shape: ${graph.soapy.config.outputShape}

  win_mul:
    module: multiply
    device: ${computeDevice}
    dataType: CF32
    input:
      factorA: ${graph.soapy.output.buffer}
      factorB: ${graph.win.output.window}

  fft:
    module: fft
    device: ${computeDevice}
    dataType: CF32
    config:
      forward: true
    input:
      buffer: ${graph.win_mul.output.product}

  amp:
    module: amplitude
    device: ${computeDevice}
    inputDataType: CF32
    outputDataType: F32
    input:
      buffer: ${graph.fft.output.buffer}

  scl:
    module: scale-view
    device: ${computeDevice}
    dataType: F32
    config:
      range: [-100.0, 0.0]
    input:
      buffer: ${graph.amp.output.buffer}

  flt:
    module: filter
    device: ${computeDevice}
    dataType: CF32
    config:
      signalSampleRate: 2.5e6
      filterSampleRate: 200e3
      filterCenter: 0.0
      shape: ${graph.soapy.config.outputShape}
      numberOfTaps: 111
      linearFrequency: true

  flt_mul:
    module: multiply
    device: ${computeDevice}
    dataType: CF32
    input:
      factorA: ${graph.fft.output.buffer}
      factorB: ${graph.flt.output.coeffs}

  amp-2:
    module: amplitude
    device: ${computeDevice}
    inputDataType: CF32
    outputDataType: F32
    input:
      buffer: ${graph.flt_mul.output.product}

  scl-2:
    module: scale-view
    device: ${computeDevice}
    dataType: F32
    config:
      range: [-100.0, 0.0]
    input:
      buffer: ${graph.amp-2.output.buffer}

  ifft:
    module: fft
    device: ${computeDevice}
    dataType: CF32
    config:
      forward: false
      offset: 942
      size: 164
    input:
      buffer: ${graph.flt_mul.output.product}

  fm:
    module: fm
    device: ${computeDevice}
    dataType: CF32
    input:
      buffer: ${graph.ifft.output.buffer}

  fft-3:
    module: fft
    device: ${computeDevice}
    dataType: CF32
    config:
      forward: true
    input:
      buffer: ${graph.ifft.output.buffer}

  amp-3:
    module: amplitude
    device: ${computeDevice}
    inputDataType: CF32
    outputDataType: F32
    input:
      buffer: ${graph.fft-3.output.buffer}

  scl-3:
    module: scale-view
    device: ${computeDevice}
    dataType: F32
    config:
      range: [-100.0, 0.0]
    input:
      buffer: ${graph.amp-3.output.buffer}
  
  audio:
    module: audio
    device: cpu
    dataType: F32
    config:
      inSampleRate: 200e3
      outSampleRate: 48e3
    input:
      buffer: ${graph.fm.output.buffer}
  
  wtf:
      module: waterfall-view
      device: ${computeDevice}
      dataType: F32
      input:
        buffer: ${graph.scl-2.output.buffer}
      interface:
        previewEnabled: true

  wtf-2:
      module: waterfall-view
      device: ${computeDevice}
      dataType: F32
      input:
        buffer: ${graph.scl-3.output.buffer}
      interface:
        previewEnabled: true

  wtf-3:
      module: waterfall-view
      device: ${computeDevice}
      dataType: F32
      input:
        buffer: ${graph.scl.output.buffer}
      interface:
        previewEnabled: true
)";

#endif /* demo_config_h */
