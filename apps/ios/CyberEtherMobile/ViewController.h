//
//  ViewController.h
//  CyberEtherMobile
//
//  Created by Luigi Cruz on 11/30/22.
//

#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include <memory>
#include <thread>

#include <jetstream/base.hh>

using namespace Jetstream;

constexpr static Device ComputeDevice = Device::Metal;
constexpr static Device RenderDevice  = Device::Metal;
using Platform = Viewport::iOS<RenderDevice>;

@interface ViewController : UIViewController {
    CADisplayLink* timer;
    CAMetalLayer* layer;
    
    Instance instance;
}

- (void) draw;
- (void) computeThread;

@end
