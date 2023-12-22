//
//  ViewController.m
//  CyberEtherMobile
//
//  Created by Luigi Cruz on 11/30/22.
//

#import "ViewController.h"

@interface ViewController ()

@end

@implementation ViewController

//
// View Setup
//

- (void)viewDidLoad {
    [super viewDidLoad];
    
    NSLog(@"Welcome to CyberEther!");
    
    // Create new CAMetalLayer.
    layer = [[CAMetalLayer alloc] init];
    layer.frame = self.view.layer.frame;
    
    // Initialize Backend.
    if (Backend::Initialize<Device::CPU>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize CPU backend.");
        JST_CHECK_THROW(Result::ERROR);
    }
    
    if (Backend::Initialize<Device::Metal>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize Metal backend.");
        JST_CHECK_THROW(Result::ERROR);
    }
    
    // Initializing Viewport.
    Viewport::Config viewportCfg;
    viewportCfg.vsync = true;
    viewportCfg.size = {3130, 1140};
    viewportCfg.title = "CyberEther";
    JST_CHECK_THROW(instance.buildViewport<Platform>(viewportCfg,
                                                     (__bridge CA::MetalLayer*)layer));
    
    JST_CHECK_THROW(instance.buildRender<Device::Metal>({}));
    
    // Attach configured layer to view.
    [self.view.layer addSublayer:layer];
    
    [NSThread detachNewThreadSelector:@selector(computeThread) toTarget:self withObject:nil];
    
    // Add graphical thread.
    // TODO: Update this value when on Low Power Mode to conserver power.
    timer = [CADisplayLink displayLinkWithTarget:self selector:@selector(draw)];
    timer.preferredFramesPerSecond = 120;
    [timer addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSRunLoopCommonModes];
    
    // Add long press gesture recognizer.
    UILongPressGestureRecognizer *longPressRecognizer = [[UILongPressGestureRecognizer alloc] initWithTarget:self action:@selector(handleLongPress:)];
    [self.view addGestureRecognizer:longPressRecognizer];
    
    // Add Apple Pencil hover gesture recognizer.
    UIHoverGestureRecognizer *hoverRecognizer = [[UIHoverGestureRecognizer alloc] initWithTarget:self action:@selector(handleHover:)];
    [self.view addGestureRecognizer:hoverRecognizer];
}

- (void)computeThread {
    while (instance.viewport().keepRunning()) {
        const auto result = instance.compute();

        if (result == Result::SUCCESS ||
            result == Result::TIMEOUT) {
            continue;
        }

        JST_CHECK_THROW(result);
    }
}

- (void)viewDidDisappear:(BOOL)animated {
    [super viewDidDisappear:animated];
    
    ImNodes::DestroyContext();
    
    instance.destroy();
    
    NSLog(@"Goodbye from CyberEther!");
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    
    [layer setFrame:self.view.frame];
    [layer setDrawableSize:CGSize{self.view.frame.size.width * 2.0, self.view.frame.size.height * 2.0}];
}

//
// Rendering
//

-(void) draw {
    if (instance.begin() == Result::SKIP) {
        return;
    }
    JST_CHECK_THROW(instance.present());
    if (instance.end() == Result::SKIP) {
        return;
    }
}

//
// Touch Input Handling
//

-(void)updateIOWithTouchEvent:(UIEvent *)event {
    UITouch *anyTouch = event.allTouches.anyObject;
    CGPoint touchLocation = [anyTouch locationInView:self.view];
    instance.viewport().addMousePosEvent(touchLocation.x, touchLocation.y);

    BOOL hasActiveTouch = NO;
    for (UITouch *touch in event.allTouches) {
        if (touch.phase != UITouchPhaseEnded && touch.phase != UITouchPhaseCancelled) {
            hasActiveTouch = YES;
            break;
        }
    }
    instance.viewport().addMouseButtonEvent(0, hasActiveTouch);
    if (!hasActiveTouch) {
        instance.viewport().addMousePosEvent(-FLT_MAX, -FLT_MAX);
    }
}

-(void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self updateIOWithTouchEvent:event];
}

-(void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self updateIOWithTouchEvent:event];
}

-(void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self updateIOWithTouchEvent:event];
}

-(void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self updateIOWithTouchEvent:event];
}

-(void)handleHover:(UIHoverGestureRecognizer *)gestureRecognizer {
    if (gestureRecognizer.state == UIGestureRecognizerStateBegan ||
        gestureRecognizer.state == UIGestureRecognizerStateChanged) {
        CGPoint hoverLocation = [gestureRecognizer locationInView:gestureRecognizer.view];
        instance.viewport().addMousePosEvent(hoverLocation.x, hoverLocation.y);
    } else if (gestureRecognizer.state == UIGestureRecognizerStateEnded ||
               gestureRecognizer.state == UIGestureRecognizerStateCancelled) {
        instance.viewport().addMousePosEvent(-FLT_MAX, -FLT_MAX);
    }
}

-(void)handleLongPress:(UILongPressGestureRecognizer *)gestureRecognizer {
    if (gestureRecognizer.state == UIGestureRecognizerStateBegan) {
        instance.viewport().addMouseButtonEvent(1, true);
    } else if (gestureRecognizer.state == UIGestureRecognizerStateEnded || 
               gestureRecognizer.state == UIGestureRecognizerStateCancelled) {
        instance.viewport().addMouseButtonEvent(1, false);
    }
}

@end
