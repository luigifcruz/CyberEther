core_test_dependencies = [libjetstream_dep, catch2_dep]
core_test_includes = include_directories('../support')
core_test_main = '../support/catch_main.cc'

core_test_domains = {
    'application': ['application/instance_backend.cc'],
    'compositor': [
        'compositor/compositor.cc',
        'compositor/docs.cc',
        'compositor/file_picker.cc',
        'compositor/gallery.cc',
        'compositor/layout.cc',
        'compositor/metrics.cc',
        'compositor/settings_tab.cc',
        'compositor/start_tab.cc',
        'compositor/tiles.cc',
    ],
    'extensions': [
        'extensions/plugins.cc',
        'extensions/benchmark.cc',
    ],
    'flowgraph': [
        'flowgraph/api.cc',
        'flowgraph/config_edits.cc',
        'flowgraph/execution.cc',
        'flowgraph/mutation.cc',
        'flowgraph/state.cc',
    ],
    'lifecycle': ['lifecycle/block_module.cc'],
    'memory': [
        'memory/axis.cc',
        'memory/token.cc',
        'memory/snapshot.cc',
        'memory/buffer.cc',
        'memory/tensor.cc',
    ],
    'platform': [
        'platform/paths.cc',
        'platform/secrets.cc',
        'platform/storage.cc',
        'platform/thread.cc',
    ],
    'registry': ['registry/registry.cc'],
    'render': [
        'render/colormap.cc',
        'render/components.cc',
        'render/browser_wheel.cc',
        'render/input_state.cc',
        'render/present_list.cc',
        'render/transfers.cc',
        'render/surface_edits.cc',
        'render/geomap_labels.cc',
        'render/map_layers.cc',
        'render/shapes.cc',
    ],
    'sakura': [
        'sakura/chat.cc',
        'sakura/chat_composer.cc',
        'sakura/interface_controls.cc',
        'sakura/scissor.cc',
        'sakura/key_value.cc',
        'sakura/flowgraph_editor_views.cc',
        'sakura/flowgraph_node_view.cc',
        'sakura/flowgraph_pane.cc',
        'sakura/metric_progress_bar.cc',
        'sakura/node_metadata.cc',
        'sakura/component.cc',
        'sakura/card.cc',
        'sakura/controls.cc',
        'sakura/tip_banner.cc',
        'sakura/voice.cc',
        'sakura/dock.cc',
        'sakura/geometry.cc',
        'sakura/idle_input.cc',
        'sakura/input.cc',
        'sakura/input_ownership.cc',
        'sakura/modal_input.cc',
        'sakura/config_identity.cc',
        'sakura/popup_lifetime.cc',
        'sakura/layout.cc',
        'sakura/node_editor.cc',
        'sakura/node_fields.cc',
        'sakura/retention.cc',
        'sakura/retention_shell.cc',
        'sakura/search.cc',
        'sakura/settings_components.cc',
        'sakura/flowgraph_card.cc',
        'sakura/pill_tabs.cc',
        'sakura/shell.cc',
        'sakura/table.cc',
        'sakura/views.cc',
        'sakura/view_components.cc',
        'sakura/sizing.cc',
        'sakura/surface_view.cc',
        'sakura/input_bridge.cc',
        'sakura/superluminal.cc',
        'sakura/syntax_highlighter.cc',
        'sakura/text_grid.cc',
        'sakura/text_markdown.cc',
        'sakura/text_metrics.cc',
        'sakura/window_metrics.cc',
    ],
    'runtime': [
        'runtime/scheduler.cc',
        'runtime/native.cc',
        'runtime/python.cc',
    ],
    'serialization': [
        'serialization/interface_format.cc',
        'serialization/conversions.cc',
        'serialization/serdes.cc',
        'serialization/hash.cc',
        'serialization/json.cc',
        'serialization/yaml.cc',
    ],
    'settings': ['settings/settings.cc'],
    'surface': ['surface/surface.cc'],
    'tools': ['tools/circular_buffer.cc'],
}

# Internal compositor and Sakura components are not exported from the Windows shared library.
if not jst_is_windows
    core_test_domains += {
        'compositor': core_test_domains['compositor'] + [
            'compositor/config_edits.cc',
            'compositor/live_markdown.cc',
            'compositor/agent_bridge.cc',
            'compositor/agent_providers.cc',
            'compositor/agent_responsiveness.cc',
        ],
    }
endif

core_test_targets = []
if jst_is_windows and cfg_lst.get('JETSTREAM_LOADER_GLFW_AVAILABLE', false)
    windows_titlebar_tests = executable(
        'jetstream-core-windows-titlebar-tests',
        ['render/windows_titlebar.cc', '../../src/viewport/glfw/tools/windows_titlebar.cc'],
        dependencies: dep_lst + [catch2_dep],
        include_directories: inc_lst,
    )
    core_test_targets += [windows_titlebar_tests]
    test('core-windows-titlebar', windows_titlebar_tests,
         suite: 'core', is_parallel: false, timeout: 60)
endif
if jst_is_macos and cfg_lst.get('JETSTREAM_LOADER_GLFW_AVAILABLE', false)
    cocoa_titlebar_tests = executable(
        'jetstream-core-cocoa-titlebar-tests',
        'render/cocoa_titlebar.mm',
        dependencies: dep_lst + [catch2_dep],
        include_directories: inc_lst + [core_test_includes],
    )
    core_test_targets += [cocoa_titlebar_tests]
    test('core-cocoa-titlebar', cocoa_titlebar_tests,
         suite: 'core', is_parallel: false, timeout: 60)
    cocoa_pointer_tests = executable(
        'jetstream-core-cocoa-pointer-tests',
        'render/cocoa_pointer.mm',
        dependencies: dep_lst + [catch2_dep],
        link_with: libjetstream,
        include_directories: inc_lst + [core_test_includes],
    )
    core_test_targets += [cocoa_pointer_tests]
    test('core-cocoa-pointer', cocoa_pointer_tests,
         suite: 'core', is_parallel: false, timeout: 60)
endif
if cfg_lst.get('JETSTREAM_RENDER_METAL_AVAILABLE', false)
    metal_program_cache_sources = ['render/metal_program_cache.cc']
    if not jst_is_browser and not jst_is_static
        metal_program_cache_sources += ['../../src/render/devices/metal/program.cc']
    endif
    core_test_domains += {
        'metal-program-cache': metal_program_cache_sources,
    }
endif
if not jst_is_browser
    core_test_domains += {
        'platform': core_test_domains['platform'] + ['platform/socket.cc'],
    }
endif
if cfg_lst.get('JETSTREAM_RENDER_METAL_AVAILABLE', false)
    core_test_domains += {
        'render': core_test_domains['render'] + ['render/metal_capture.cc'],
    }
endif
if cfg_lst.get('JETSTREAM_LOADER_GSTREAMER_AVAILABLE', false) and not jst_is_windows
    core_test_domains += {
        'application': core_test_domains['application'] + ['application/remote_stream.cc'],
    }
endif

foreach domain, sources : core_test_domains
    # The sakura suite drives Sakura internals, which are not exported from
    # the shared library on Windows.
    if domain == 'sakura' and jst_is_windows
        continue
    endif
    domain_tests = executable(
        'jetstream-core-' + domain + '-tests',
        [core_test_main] + sources,
        dependencies: core_test_dependencies,
        include_directories: core_test_includes,
        link_args: test_link_args,
        link_depends: browser_exe_link_depends,
        build_by_default: test_build_by_default,
        cpp_args: domain == 'runtime' ? [
            '-DJETSTREAM_PYTHON_TEST_WHEEL="' + (meson.current_source_dir() /
                'runtime/fixtures/cyberether_cache_fixture-1.0-py3-none-any.whl') + '"',
        ] : [],
    )
    core_test_targets += [domain_tests]
    if jst_is_browser
        test('core-' + domain, node,
             args: [files('../browser/run.cjs'), domain_tests.full_path()],
             depends: domain_tests,
             suite: ['core', 'wasm'], is_parallel: false, timeout: 180)
    else
        test('core-' + domain, domain_tests,
             suite: 'core', is_parallel: false, timeout: 180)
    endif
endforeach

# TODO: Re-enable after CLI output capture supports separate Windows static CRTs.
if not jst_is_ios and not jst_is_android and not jst_is_windows
    core_integration_tests = executable(
        'jetstream-core-integration-tests',
        'integration/cli.cc',
        dependencies: core_test_dependencies,
        include_directories: core_test_includes,
        link_args: test_link_args,
        link_depends: browser_exe_link_depends,
        build_by_default: test_build_by_default,
    )
    core_test_targets += [core_integration_tests]
    if jst_is_browser
        test('core-integration', node,
             args: [files('../browser/run.cjs'), core_integration_tests.full_path(), '--timeout', '45000'],
             depends: core_integration_tests,
             suite: ['core', 'wasm'], is_parallel: false, timeout: 60)
    else
        test('core-integration', core_integration_tests,
             suite: 'core', is_parallel: false, timeout: 60)
    endif
endif

alias_target('jetstream-core-tests', core_test_targets)
