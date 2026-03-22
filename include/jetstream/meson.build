configure_file(
   input : 'config.hh.in',
   output : 'config.hh',
   configuration : cfg_lst,
   install : not jst_is_wheel,
   install_dir : get_option('includedir') + '/jetstream'
)
