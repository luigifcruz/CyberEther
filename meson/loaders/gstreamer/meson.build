deps = [
    dependency('gstreamer-1.0', required: false),
    dependency('gstreamer-app-1.0', required: false),
    dependency('gstreamer-video-1.0', required: false),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_GSTREAMER_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Gstreamer': all_deps_found}
