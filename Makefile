NAME = edge-impulse-standalone

EI_SDK?=edge-impulse-sdk

CC_VERSION = $(shell $(CC) --version)

CFLAGS += -g -DALLOC -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DEI_CLASSIFIER_TFLITE_ENABLE_CMSIS_NN=1 -D__ARM_FEATURE_DSP=1 -DARM_MATH_DSP=1 -D__GNUC_PYTHON__=1
ifeq (,$(findstring x86,$(CC_VERSION)))
	# non-x86 systems
	CFLAGS += -DTF_LITE_DISABLE_X86_NEON -DARM_MATH_NEON=1 -mfloat-abi=hard -mfpu=neon
endif

CFLAGS += -I./
CFLAGS += -I./source/
CFLAGS += -I${EI_SDK}/
CFLAGS += -I${EI_SDK}/tensorflow
CFLAGS += -I${EI_SDK}/third_party
CFLAGS += -I${EI_SDK}/third_party/flatbuffers
CFLAGS += -I${EI_SDK}/third_party/flatbuffers/include
CFLAGS += -I${EI_SDK}/third_party/flatbuffers/include/flatbuffers
CFLAGS += -I${EI_SDK}/third_party/gemmlowp/
CFLAGS += -I${EI_SDK}/third_party/gemmlowp/fixedpoint
CFLAGS += -I${EI_SDK}/third_party/gemmlowp/internal
CFLAGS += -I${EI_SDK}/third_party/ruy
CFLAGS += -I${EI_SDK}/anomaly
CFLAGS += -I${EI_SDK}/classifier
CFLAGS += -I${EI_SDK}/dsp
CFLAGS += -I${EI_SDK}/dsp/kissfft
CFLAGS += -I${EI_SDK}/porting
CFLAGS += -I${EI_SDK}/CMSIS/Core/Include/
CFLAGS += -I${EI_SDK}/CMSIS/NN/Include/
CFLAGS += -I${EI_SDK}/CMSIS/DSP/Include/
CFLAGS += -I${EI_SDK}/CMSIS/DSP/PrivateInclude/
CFLAGS += -I./model-parameters
CFLAGS += -I./tflite-model
CXXFLAGS += -std=c++11

CSOURCES = ${EI_SDK}/tensorflow/lite/c/common.c $(wildcard ${EI_SDK}/CMSIS/NN/Source/ActivationFunctions/*.c) $(wildcard ${EI_SDK}/CMSIS/NN/Source/BasicMathFunctions/*.c) $(wildcard ${EI_SDK}/CMSIS/NN/Source/ConcatenationFunctions/*.c) $(wildcard ${EI_SDK}/CMSIS/NN/Source/ConvolutionFunctions/*.c) $(wildcard ${EI_SDK}/CMSIS/NN/Source/FullyConnectedFunctions/*.c) $(wildcard ${EI_SDK}/CMSIS/NN/Source/NNSupportFunctions/*.c) $(wildcard ${EI_SDK}/CMSIS/NN/Source/PoolingFunctions/*.c) $(wildcard ${EI_SDK}/CMSIS/NN/Source/ReshapeFunctions/*.c) $(wildcard ${EI_SDK}/CMSIS/NN/Source/SoftmaxFunctions/*.c)
CXXSOURCES = $(wildcard source/*.cpp) $(wildcard ${EI_SDK}/dsp/kissfft/*.cpp) $(wildcard ${EI_SDK}/dsp/dct/*.cpp) $(wildcard ./${EI_SDK}/dsp/memory.cpp) $(wildcard ${EI_SDK}/porting/posix/*.c*) $(wildcard tflite-model/*.cpp)
CCSOURCES = $(wildcard ${EI_SDK}/tensorflow/lite/kernels/*.cc) $(wildcard ${EI_SDK}/tensorflow/lite/kernels/internal/*.cc) $(wildcard ${EI_SDK}/tensorflow/lite/micro/kernels/*.cc) $(wildcard ${EI_SDK}/tensorflow/lite/micro/*.cc) $(wildcard ${EI_SDK}/tensorflow/lite/micro/memory_planner/*.cc) $(wildcard ${EI_SDK}/tensorflow/lite/core/api/*.cc)

COBJECTS := $(patsubst %.c,%.o,$(CSOURCES))
CXXOBJECTS := $(patsubst %.cpp,%.o,$(CXXSOURCES))
CCOBJECTS := $(patsubst %.cc,%.o,$(CCSOURCES))

all: app

.PHONY: app clean

$(COBJECTS) : %.o : %.c
$(CXXOBJECTS) : %.o : %.cpp
$(CCOBJECTS) : %.o : %.cc

%.o: %.c
	$(CC) -std=gnu11 $(CFLAGS) -c $^ -o $@

%.o: %.cc
	$(CXX) $(CFLAGS) $(CXXFLAGS) -c $^ -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CXXFLAGS) -c $^ -o $@

app: $(COBJECTS) $(CXXOBJECTS) $(CCOBJECTS)
	mkdir -p build
	$(CXX) -ldl $(COBJECTS) $(CXXOBJECTS) $(CCOBJECTS) -o build/$(NAME)

clean:
	rm -f $(COBJECTS)
	rm -f $(CCOBJECTS)
	rm -f $(CXXOBJECTS)
	rm -f build/$(NAME)
