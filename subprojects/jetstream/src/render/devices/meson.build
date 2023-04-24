sum_lst = {}

subdir('metal')

render_found = true
foreach i : sum_lst.keys()
    render_found = render_found and sum_lst[i]
endforeach

if not render_found
    error('No render found for this platform.')
endif

summary(sum_lst, section: 'Render (Present)', bool_yn: true)
