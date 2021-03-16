cfg_lst.set('version', meson.project_version())

configure_file(input : 'render_config.hpp.in',
               output : 'render_config.hpp',
               configuration : cfg_lst,
               install : true,
               install_dir : get_option('includedir') + '/render')

install_subdir('render', install_dir : get_option('includedir'))