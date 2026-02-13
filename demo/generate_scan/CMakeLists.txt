add_executable(generate_scan main.cpp)
set_target_properties(generate_scan PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

target_link_libraries(generate_scan PUBLIC vortex::vortex)
