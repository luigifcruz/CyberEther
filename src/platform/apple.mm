#include "apple.hh"

#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>

#ifdef JST_OS_IOS
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#endif

namespace Jetstream::Platform {

Result OpenUrl(const std::string& url) {
    NSString* nsUrl = [NSString stringWithUTF8String:url.c_str()];
    NSURL* urlObj = [NSURL URLWithString:nsUrl];

    if (!urlObj) {
        JST_ERROR("Cannot open URL because it's invalid.");
        return Result::ERROR;
    }

#ifdef JST_OS_IOS
    if ([[UIApplication sharedApplication] canOpenURL:urlObj]) {
        [[UIApplication sharedApplication] openURL:urlObj options:@{} completionHandler:nil];
    } else {
        JST_ERROR("Cannot open URL.");
        return Result::ERROR;
    }
#else
    if (![[NSWorkspace sharedWorkspace] openURL:urlObj]) {
        JST_ERROR("Cannot open URL.");
        return Result::ERROR;
    }
#endif

    return Result::SUCCESS;
}

Result PickFile(std::string& path) {
    __block Result result = Result::ERROR;

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    dispatch_async(dispatch_get_main_queue(), ^{
#ifdef JST_OS_MAC
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];

        UTType* yamlType = [UTType typeWithFilenameExtension:@"yaml"];
        UTType* ymlType = [UTType typeWithFilenameExtension:@"yml"];
        [panel setAllowedContentTypes:@[yamlType, ymlType]];

        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [[panel URLs] objectAtIndex:0];
            NSString* filePath = [url path];
            path = std::string([filePath UTF8String]);
            result = Result::SUCCESS;
        } else {
            JST_ERROR("Cannot pick file.");
            result = Result::ERROR;
        }
#endif
        dispatch_semaphore_signal(semaphore);
    });

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

    return result;
}

Result PickFolder(std::string& path) {
    __block Result result = Result::ERROR;

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    dispatch_async(dispatch_get_main_queue(), ^{
#ifdef JST_OS_MAC
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setCanChooseFiles:NO];
        [panel setCanChooseDirectories:YES];
        [panel setAllowsMultipleSelection:NO];

        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [[panel URLs] objectAtIndex:0];
            NSString* folderPath = [url path];
            path = std::string([folderPath UTF8String]);
            result = Result::SUCCESS;
        } else {
            JST_ERROR("Cannot pick folder.");
            result = Result::ERROR;
        }
#endif
        dispatch_semaphore_signal(semaphore);
    });

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

    return result;
}

Result SaveFile(std::string& path) {
    __block Result result = Result::ERROR;

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    dispatch_async(dispatch_get_main_queue(), ^{
#ifdef JST_OS_MAC
        NSSavePanel* panel = [NSSavePanel savePanel];

        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [panel URL];
            NSString* filePath = [url path];
            path = std::string([filePath UTF8String]);
            result = Result::SUCCESS;
        } else {
            JST_ERROR("Cannot save file.");
            result = Result::ERROR;
        }
#endif
        dispatch_semaphore_signal(semaphore);
    });

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

    return result;
}

}  // namespace Jetstream::Platform
