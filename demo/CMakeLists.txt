if(WITH_REFLEXXES)
	add_subdirectory(generate_scan)
endif()

if(WITH_CUDA AND WITH_DAQMX AND WITH_ALAZAR)
	add_subdirectory(engine)
endif()
