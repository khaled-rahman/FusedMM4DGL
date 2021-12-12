file(REMOVE_RECURSE
  "libfusedmm.pdb"
  "libfusedmm.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/fusedmm.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
