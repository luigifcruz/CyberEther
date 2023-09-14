configure_file(
   input : 'jetstream/config.hh.in',
   output : 'jetstream_config.hh',
   configuration : cfg_lst,
   install : true,
   install_dir : get_option('includedir') + '/jetstream'
)

install_subdir('jetstream', install_dir: get_option('includedir'))
