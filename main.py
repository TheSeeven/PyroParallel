import PyroParallel as Pyro
import numpy as np
import time
import os

os.environ["PYOPENCL_DISABLE_WATCHDOG"] = "1"

Framework_all = Pyro.PyroParallel(verbose=True,
                                  exclude_FPGA=True,
                                  exclude_others=True,
                                  emulation=False)

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

NUMBER_OF_TESTS = 200
TEST_DATA = [
    np.full((3000, 3000, 3), 255, dtype=np.uint8)
    for _ in range(NUMBER_OF_TESTS)
]
for _ in range(1, 11):
    print("RUN: " + str(_))
    time.sleep(2)
    start = time.time()
    Framework_only_CPU.grayscale(TEST_DATA)
    end = time.time()
    print("CPU: {0}".format(str((end - start) * 1000)), )
    time.sleep(2)
    start = time.time()
    Framework_only_GPUS.grayscale(TEST_DATA)
    end = time.time()
    print("GPUS: {0}".format(str((end - start) * 1000)), )
    time.sleep(2)
    start = time.time()
    Framework_all.grayscale(TEST_DATA)
    end = time.time()
    print("All devices: {0}".format(str((end - start) * 1000)), )
    time.sleep(2)
    start = time.time()
    Framework_only_iGPU.grayscale(TEST_DATA)
    end = time.time()
    print("iGPU: {0}".format(str((end - start) * 1000)), )
    time.sleep(2)
    print("END RUN:{0}\n\n".format(str(_)))
print("done..")
