import PyroParallel as Pyro
import numpy as np
import time
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
### TESTING PANNEL
TEST_ALL = 0
TEST_ONLY_CPU = 1
TEST_ONLY_GPUS = 2
TEST_ONLY_iGPU = 3

PROCESSING_MODE = Pyro.PLATFORM_MODE
TEST_MODE = 3

### TESTING PANNEL

Framework_all = Pyro.PyroParallel(verbose=True,
                                  exclude_FPGA=True,
                                  exclude_others=True,
                                  emulation=False,
                                  processing_mode=PROCESSING_MODE,
                                  CHUNK_PROCESSING_SIZE=1)
Framework_only_CPU = Pyro.PyroParallel(verbose=True,
                                       exclude_FPGA=True,
                                       exclude_others=True,
                                       emulation=False,
                                       processing_mode=PROCESSING_MODE,
                                       exclude_GPU=True)

Framework_only_GPUS = Pyro.PyroParallel(verbose=True,
                                        exclude_FPGA=True,
                                        exclude_others=True,
                                        emulation=False,
                                        processing_mode=PROCESSING_MODE,
                                        exclude_CPU=True)

Framework_only_iGPU = Pyro.PyroParallel(verbose=True,
                                        exclude_FPGA=True,
                                        exclude_others=True,
                                        emulation=False,
                                        processing_mode=PROCESSING_MODE,
                                        exclude_CPU=True)

Framework_only_iGPU.opencl_devices[0] = Framework_only_iGPU.opencl_devices[1]
Framework_only_iGPU.opencl_devices = Framework_only_iGPU.opencl_devices[:-1]
Framework_only_iGPU.opencl_platforms[0] = Framework_only_iGPU.opencl_platforms[
    1]
Framework_only_iGPU.opencl_platforms = Framework_only_iGPU.opencl_platforms[:
                                                                            -1]
Framework_all.benchmark_api()
Framework_only_CPU.benchmark_api()
Framework_only_GPUS.benchmark_api()
Framework_only_iGPU.benchmark_api()

### TEST DATA
NUMBER_OF_TESTS = 20
TEST_DATA = [
    np.full((3000, 3000, 3), 255, dtype=np.uint8)
    for _ in range(NUMBER_OF_TESTS)
]

TEST_FP32_A = np.full(100, 1.0, dtype=np.float32)
TEST_FP32_B = np.full(100, 2.0, dtype=np.float32)

TEST_FP64_A = np.full(100, 1.0, dtype=np.float64)
TEST_FP64_B = np.full(100, 2.0, dtype=np.float64)
input_image = np.array(Image.open("./sample_data/5120x2880_1_NO_HDR.bmp"))
EDGE_DETECTION_SAMPLE_IMAGE = Framework_all.grayscale(
    [input_image for x in range(50)], autosave=True)

### TEST DATA

### FUNCTIONALITY TESTING

# x = Framework_all.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
#                                  120,
#                                  autosave=True)
x = Framework_only_CPU.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
                                      120,
                                      autosave=True)
x = Framework_only_GPUS.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
                                       120,
                                       autosave=True)
x = Framework_only_iGPU.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
                                       120,
                                       autosave=True)
quit()
if TEST_MODE == TEST_ALL:
    # grayscale tests
    Framework_all.grayscale(TEST_DATA, autosave=True)

    # fp32 tests
    Framework_all.operation([TEST_FP32_A for x in range(100)],
                            [TEST_FP32_B for x in range(100)],
                            Pyro.ADDITION,
                            Pyro.FP32,
                            autosave=True)
    Framework_all.operation([TEST_FP32_A for x in range(100)],
                            [TEST_FP32_B for x in range(100)],
                            Pyro.SUBTRACT,
                            Pyro.FP32,
                            autosave=True)
    Framework_all.operation([TEST_FP32_A for x in range(100)],
                            [TEST_FP32_B for x in range(100)],
                            Pyro.MULTIPLY,
                            Pyro.FP32,
                            autosave=True)
    Framework_all.operation([TEST_FP32_A for x in range(100)],
                            [TEST_FP32_B for x in range(100)],
                            Pyro.DIVISION,
                            Pyro.FP32,
                            autosave=True)

    # fp64 tests
    Framework_all.operation([TEST_FP64_A for x in range(100)],
                            [TEST_FP64_B for x in range(100)],
                            Pyro.ADDITION,
                            Pyro.FP64,
                            autosave=True)
    Framework_all.operation([TEST_FP64_A for x in range(100)],
                            [TEST_FP64_B for x in range(100)],
                            Pyro.SUBTRACT,
                            Pyro.FP64,
                            autosave=True)
    Framework_all.operation([TEST_FP64_A for x in range(100)],
                            [TEST_FP64_B for x in range(100)],
                            Pyro.MULTIPLY,
                            Pyro.FP64,
                            autosave=True)
    Framework_all.operation([TEST_FP64_A for x in range(100)],
                            [TEST_FP64_B for x in range(100)],
                            Pyro.DIVISION,
                            Pyro.FP64,
                            autosave=True)

if TEST_MODE == TEST_ONLY_CPU:
    # grayscale tests
    Framework_only_CPU.grayscale(TEST_DATA, autosave=True)

    # fp32 tests
    Framework_only_CPU.operation([TEST_FP32_A for x in range(100)],
                                 [TEST_FP32_B for x in range(100)],
                                 Pyro.ADDITION,
                                 Pyro.FP32,
                                 autosave=True)
    Framework_only_CPU.operation([TEST_FP32_A for x in range(100)],
                                 [TEST_FP32_B for x in range(100)],
                                 Pyro.SUBTRACT,
                                 Pyro.FP32,
                                 autosave=True)
    Framework_only_CPU.operation([TEST_FP32_A for x in range(100)],
                                 [TEST_FP32_B for x in range(100)],
                                 Pyro.MULTIPLY,
                                 Pyro.FP32,
                                 autosave=True)
    Framework_only_CPU.operation([TEST_FP32_A for x in range(100)],
                                 [TEST_FP32_B for x in range(100)],
                                 Pyro.DIVISION,
                                 Pyro.FP32,
                                 autosave=True)

    # fp64 tests
    Framework_only_CPU.operation([TEST_FP64_A for x in range(100)],
                                 [TEST_FP64_B for x in range(100)],
                                 Pyro.ADDITION,
                                 Pyro.FP64,
                                 autosave=True)
    Framework_only_CPU.operation([TEST_FP64_A for x in range(100)],
                                 [TEST_FP64_B for x in range(100)],
                                 Pyro.SUBTRACT,
                                 Pyro.FP64,
                                 autosave=True)
    Framework_only_CPU.operation([TEST_FP64_A for x in range(100)],
                                 [TEST_FP64_B for x in range(100)],
                                 Pyro.MULTIPLY,
                                 Pyro.FP64,
                                 autosave=True)
    Framework_only_CPU.operation([TEST_FP64_A for x in range(100)],
                                 [TEST_FP64_B for x in range(100)],
                                 Pyro.DIVISION,
                                 Pyro.FP64,
                                 autosave=True)

if TEST_MODE == TEST_ONLY_GPUS:
    # grayscale tests
    Framework_only_GPUS.grayscale(TEST_DATA, autosave=True)

    # fp32 tests
    Framework_only_GPUS.operation([TEST_FP32_A for x in range(100)],
                                  [TEST_FP32_B for x in range(100)],
                                  Pyro.ADDITION,
                                  Pyro.FP32,
                                  autosave=True)
    Framework_only_GPUS.operation([TEST_FP32_A for x in range(100)],
                                  [TEST_FP32_B for x in range(100)],
                                  Pyro.SUBTRACT,
                                  Pyro.FP32,
                                  autosave=True)
    Framework_only_GPUS.operation([TEST_FP32_A for x in range(100)],
                                  [TEST_FP32_B for x in range(100)],
                                  Pyro.MULTIPLY,
                                  Pyro.FP32,
                                  autosave=True)
    Framework_only_GPUS.operation([TEST_FP32_A for x in range(100)],
                                  [TEST_FP32_B for x in range(100)],
                                  Pyro.DIVISION,
                                  Pyro.FP32,
                                  autosave=True)

    # fp64 tests
    Framework_only_GPUS.operation([TEST_FP64_A for x in range(100)],
                                  [TEST_FP64_B for x in range(100)],
                                  Pyro.ADDITION,
                                  Pyro.FP64,
                                  autosave=True)
    Framework_only_GPUS.operation([TEST_FP64_A for x in range(100)],
                                  [TEST_FP64_B for x in range(100)],
                                  Pyro.SUBTRACT,
                                  Pyro.FP64,
                                  autosave=True)
    Framework_only_GPUS.operation([TEST_FP64_A for x in range(100)],
                                  [TEST_FP64_B for x in range(100)],
                                  Pyro.MULTIPLY,
                                  Pyro.FP64,
                                  autosave=True)
    Framework_only_GPUS.operation([TEST_FP64_A for x in range(100)],
                                  [TEST_FP64_B for x in range(100)],
                                  Pyro.DIVISION,
                                  Pyro.FP64,
                                  autosave=True)

if TEST_MODE == TEST_ONLY_iGPU:
    # grayscale tests
    Framework_only_iGPU.grayscale(TEST_DATA, autosave=True)

    # fp32 tests
    Framework_only_iGPU.operation([TEST_FP32_A for x in range(100)],
                                  [TEST_FP32_B for x in range(100)],
                                  Pyro.ADDITION,
                                  Pyro.FP32,
                                  autosave=True)
    Framework_only_iGPU.operation([TEST_FP32_A for x in range(100)],
                                  [TEST_FP32_B for x in range(100)],
                                  Pyro.SUBTRACT,
                                  Pyro.FP32,
                                  autosave=True)
    Framework_only_iGPU.operation([TEST_FP32_A for x in range(100)],
                                  [TEST_FP32_B for x in range(100)],
                                  Pyro.MULTIPLY,
                                  Pyro.FP32,
                                  autosave=True)
    Framework_only_iGPU.operation([TEST_FP32_A for x in range(100)],
                                  [TEST_FP32_B for x in range(100)],
                                  Pyro.DIVISION,
                                  Pyro.FP32,
                                  autosave=True)

    # fp64 tests
    Framework_only_iGPU.operation([TEST_FP64_A for x in range(100)],
                                  [TEST_FP64_B for x in range(100)],
                                  Pyro.ADDITION,
                                  Pyro.FP64,
                                  autosave=True)
    Framework_only_iGPU.operation([TEST_FP64_A for x in range(100)],
                                  [TEST_FP64_B for x in range(100)],
                                  Pyro.SUBTRACT,
                                  Pyro.FP64,
                                  autosave=True)
    Framework_only_iGPU.operation([TEST_FP64_A for x in range(100)],
                                  [TEST_FP64_B for x in range(100)],
                                  Pyro.MULTIPLY,
                                  Pyro.FP64,
                                  autosave=True)
    Framework_only_iGPU.operation([TEST_FP64_A for x in range(100)],
                                  [TEST_FP64_B for x in range(100)],
                                  Pyro.DIVISION,
                                  Pyro.FP64,
                                  autosave=True)

### FUNCTIONALITY TESTING

# grayscale performance
# for _ in range(1, 11):
#     print("RUN: " + str(_))
#     time.sleep(2)
#     start = time.time()
#     Framework_only_CPU.grayscale(TEST_DATA)
#     end = time.time()
#     print("CPU: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_only_GPUS.grayscale(TEST_DATA)
#     end = time.time()
#     print("GPUS: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_all.grayscale(TEST_DATA)
#     end = time.time()
#     print("All devices: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_only_iGPU.grayscale(TEST_DATA)
#     end = time.time()
#     print("iGPU: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     print("END RUN:{0}\n\n".format(str(_)))

# FP32 performance
# for _ in range(1, 1):
#     print("RUN: " + str(_))
#     time.sleep(0)
#     start = time.time()
#     Framework_all.operation_fp32(TEST_DATA, TEST_DATA, Pyro.ADD)
#     end = time.time()
#     print("CPU: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_all.operation_fp32(TEST_DATA, TEST_DATA, Pyro.SUBTRACT)
#     end = time.time()
#     print("GPUS: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_all.operation_fp32(TEST_DATA, TEST_DATA, Pyro.DIVIDE)
#     end = time.time()
#     print("All devices: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_all.operation_fp32(TEST_DATA, TEST_DATA, Pyro.MULTIPLY)
#     end = time.time()
#     print("iGPU: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     print("END RUN:{0}\n\n".format(str(_)))

# FP64 performance

# for _ in range(1, 1):
#     print("RUN: " + str(_))
#     time.sleep(2)
#     start = time.time()
#     Framework_all.operation_fp64(TEST_DATA, TEST_DATA, Pyro.ADD)
#     end = time.time()
#     print("CPU: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_all.operation_fp64(TEST_DATA, TEST_DATA, Pyro.SUBTRACT)
#     end = time.time()
#     print("GPUS: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_all.operation_fp64(TEST_DATA, TEST_DATA, Pyro.DIVIDE)
#     end = time.time()
#     print("All devices: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     start = time.time()
#     Framework_all.operation_fp64(TEST_DATA, TEST_DATA, Pyro.MULTIPLY)
#     end = time.time()
#     print("iGPU: {0}".format(str((end - start) * 1000)), )
#     time.sleep(2)
#     print("END RUN:{0}\n\n".format(str(_)))

print("done..")
