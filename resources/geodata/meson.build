if py.found()
    python_with_modules = pymod.find_installation(
        py.full_path(),
        required : true,
        modules : ['numpy', 'mapbox_earcut'],
    )

    triangulate_list = ','.join([
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
        'ne_10m_admin_0_countries',
        'ne_10m_urban_areas',
        'ne_10m_lakes',
    ])

    line_segment_list = ','.join([
        'ne_10m_coastline',
        'ne_10m_admin_0_boundary_lines_land',
        'ne_10m_admin_1_states_provinces_lines',
    ])

    styled_line_segment_list = ','.join([
        'ne_10m_admin_0_boundary_lines_disputed_areas',
        'ne_10m_rivers_lake_centerlines',
    ])

    geographic_lines_split_list = ','.join([
        'ne_10m_geographic_lines',
    ])

    label_point_list = ','.join([
        'ne_10m_populated_places_simple',
        'ne_10m_admin_0_countries',
        'ne_10m_admin_1_states_provinces',
        'ne_10m_airports',
        'ne_10m_geography_marine_polys',
        'ne_10m_geography_regions_elevation_points',
        'ne_10m_geography_regions_polys',
        'ne_10m_rivers_lake_centerlines',
        'ne_10m_lakes',
    ])

    geodata_dir = meson.current_source_dir()

    geodata_files = [
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
        'ne_10m_lakes.geojson',
        'ne_10m_rivers_lake_centerlines.geojson',
        'ne_10m_urban_areas.geojson',
        'ne_10m_coastline.geojson',
        'ne_10m_admin_0_boundary_lines_land.geojson',
        'ne_10m_admin_0_boundary_lines_disputed_areas.geojson',
        'ne_10m_admin_1_states_provinces_lines.geojson',
        'ne_10m_geographic_lines.geojson',
        'ne_10m_populated_places_simple.geojson',
        'ne_10m_admin_0_countries.geojson',
        'ne_10m_admin_1_states_provinces.geojson',
        'ne_10m_airports.geojson',
        'ne_10m_geography_marine_polys.geojson',
        'ne_10m_geography_regions_elevation_points.geojson',
        'ne_10m_geography_regions_polys.geojson',
    ]

    geodata_paths = []
    foreach f : geodata_files
        geodata_paths += geodata_dir / f
    endforeach

    geodata = custom_target(
        'geodata',
        output : ['geodata.hh'],
        command : [python_with_modules, files('parser.py'), meson.project_build_root(), '@OUTPUT@',
                   '--triangulate=' + triangulate_list,
                   '--line-segments=' + line_segment_list,
                   '--styled-line-segments=' + styled_line_segment_list,
                   '--geographic-lines-split=' + geographic_lines_split_list,
                   '--label-points=' + label_point_list] + geodata_paths,
    )

    dep_lst += declare_dependency(
        sources : geodata,
    )
endif
