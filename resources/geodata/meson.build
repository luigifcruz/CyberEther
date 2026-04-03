if py.found()
    python_with_modules = pymod.find_installation(
        py.full_path(),
        required : true,
        modules : ['numpy', 'mapbox_earcut'],
    )

    triangulate_list = ','.join([
        'ne_10m_bathymetry_L_0',
        'ne_10m_bathymetry_K_200',
        'ne_10m_bathymetry_J_1000',
        'ne_10m_bathymetry_I_2000',
        'ne_10m_bathymetry_H_3000',
        'ne_10m_bathymetry_G_4000',
        'ne_10m_bathymetry_F_5000',
        'ne_10m_bathymetry_E_6000',
        'ne_10m_bathymetry_D_7000',
        'ne_10m_bathymetry_C_8000',
        'ne_10m_bathymetry_B_9000',
        'ne_10m_bathymetry_A_10000',
        'ne_10m_land',
        'ne_10m_urban_areas',
        'ne_10m_lakes',
    ])

    geodata_dir = meson.current_source_dir()

    geodata_files = [
        'ne_10m_bathymetry_L_0.geojson',
        'ne_10m_bathymetry_K_200.geojson',
        'ne_10m_bathymetry_J_1000.geojson',
        'ne_10m_bathymetry_I_2000.geojson',
        'ne_10m_bathymetry_H_3000.geojson',
        'ne_10m_bathymetry_G_4000.geojson',
        'ne_10m_bathymetry_F_5000.geojson',
        'ne_10m_bathymetry_E_6000.geojson',
        'ne_10m_bathymetry_D_7000.geojson',
        'ne_10m_bathymetry_C_8000.geojson',
        'ne_10m_bathymetry_B_9000.geojson',
        'ne_10m_bathymetry_A_10000.geojson',
        'ne_10m_land.geojson',
        'ne_10m_lakes.geojson',
        'ne_10m_rivers_lake_centerlines.geojson',
        'ne_10m_urban_areas.geojson',
        'ne_10m_coastline.geojson',
        'ne_10m_admin_0_boundary_lines_land.geojson',
        'ne_10m_admin_1_states_provinces_lines.geojson',
        'ne_10m_populated_places_simple.geojson',
    ]

    geodata_paths = []
    foreach f : geodata_files
        geodata_paths += geodata_dir / f
    endforeach

    geodata = custom_target(
        'geodata',
        output : ['geodata.hh'],
        command : [python_with_modules, files('parser.py'), meson.project_build_root(), '@OUTPUT@',
                   '--triangulate=' + triangulate_list] + geodata_paths,
    )

    dep_lst += declare_dependency(
        sources : geodata,
    )

    dep_lst += dependency('zlib', required: true, fallback: ['zlib', 'zlib_dep'])
endif
