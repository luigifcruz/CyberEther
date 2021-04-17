cfg_lst.set('version', meson.project_version())

configure_file(input : 'spectrum/config.in',
               output : 'spectrum_config.hpp',
               configuration : cfg_lst,
               install : true,
               install_dir : get_option('includedir') + '/spectrum')

install_subdir('spectrum', install_dir : get_option('includedir'))
