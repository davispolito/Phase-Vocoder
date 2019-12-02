cuda_add_library(karnel
	"kernel.cu"
	"kernel.h"
	OPTIONS -arch=sm_61
	) 
