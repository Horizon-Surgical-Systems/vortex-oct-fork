add_executable(engine main.cpp)
set_target_properties(engine PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

target_link_libraries(engine PUBLIC vortex::vortex)

find_package(lyra CONFIG REQUIRED)
target_link_libraries(engine PUBLIC bfg::lyra)
