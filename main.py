import PyroParallel as Pyro
import numpy as np
import time
import os

Framework_all = Pyro.PyroParallel(verbose=True,
                                  exclude_FPGA=True,
                                  exclude_others=True,
                                  emulation=False,
                                  CHUNKCHUNK_PROCESSING_SIZE=4)
Framework_only_CPU = Pyro.PyroParallel(verbose=True,
                                       exclude_FPGA=True,
                                       exclude_others=True,
                                       emulation=False,
                                       exclude_GPU=True)

Framework_only_GPUS = Pyro.PyroParallel(verbose=True,
                                        exclude_FPGA=True,
                                        exclude_others=True,
                                        emulation=False,
                                        exclude_CPU=True)

Framework_only_iGPU = Pyro.PyroParallel(verbose=True,
                                        exclude_FPGA=True,
                                        exclude_others=True,
                                        emulation=False,
                                        exclude_CPU=True)

Framework_only_iGPU.opencl_devices[0] = Framework_only_iGPU.opencl_devices[1]
Framework_only_iGPU.opencl_devices = Framework_only_iGPU.opencl_devices[:-1]

Framework_all.benchmark_api()
Framework_only_CPU.benchmark_api()
Framework_only_GPUS.benchmark_api()
Framework_only_iGPU.benchmark_api()

NUMBER_OF_TESTS = 100
TEST_DATA = [
    np.full((3000, 3000, 3), 255, dtype=np.uint8)
    for _ in range(NUMBER_OF_TESTS)
]

TEST_FP32_A = np.full(100, 1.0, dtype=np.float32)
TEST_FP32_B = np.full(100, 2.0, dtype=np.float32)

TEST_FP64_A = np.full(100, 1.0, dtype=np.float64)
TEST_FP64_B = np.full(100, 2.0, dtype=np.float64)
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
x = 10
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
