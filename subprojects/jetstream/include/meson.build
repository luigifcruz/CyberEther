cfg_lst.set('version', meson.project_version())

configure_file(input : 'jetstream/config.hpp.in',
               output : 'jetstream_config.hpp',
               configuration : cfg_lst,
               install : true,
               install_dir : get_option('includedir') + '/jetstream')

install_subdir('jetstream', install_dir : get_option('includedir'))
