cfg_lst.set('version', meson.project_version())

configure_file(input : 'jstcore/config.hpp.in',
               output : 'jstcore_config.hpp',
               configuration : cfg_lst,
               install : true,
               install_dir : get_option('includedir') + '/jstcore')

install_subdir('jstcore', install_dir : get_option('includedir'))
