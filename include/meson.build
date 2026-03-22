subdir('jetstream')

if not jst_is_static and \
   not jst_is_wheel
   install_subdir(
      'jetstream',
      install_dir: get_option('includedir'),
      exclude_files: [
         'config.hh.in',
         'meson.build',
      ],
   )
endif
