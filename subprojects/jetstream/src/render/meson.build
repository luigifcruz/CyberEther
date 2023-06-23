sum_lst = {}

subdir('metal')
subdir('vulkan')
subdir('tools')
subdir('base')

some_render_found = false
foreach x_render : sum_lst.keys()
    some_render_found = some_render_found or sum_lst.get(x_render, false)
endforeach

if not some_render_found
    # TODO: Revert.
    # error('Failed to find a compatible render. Check your dependencies and try again.')
endif

summary(sum_lst, section: 'Render (Present)', bool_yn: true)
