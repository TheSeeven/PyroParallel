import PyroParallel as Pyro
import numpy as np
import time

Framework_all = Pyro.PyroParallel(verbose=True,
                                  exclude_FPGA=True,
                                  exclude_others=True,
                                  emulation=False)

Framework_only_CPU = Pyro.PyroParallel(verbose=True,
                                       exclude_FPGA=True,
                                       exclude_others=True,
                                       emulation=False,
                                       exclude_GPU=True)

Framework_only_GPU = Pyro.PyroParallel(verbose=True,
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
Framework_only_GPU.benchmark_api()
Framework_only_iGPU.benchmark_api()

NUMBER_OF_TESTS = 10000
TEST_DATA = [
    np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)
    for _ in range(NUMBER_OF_TESTS)
]

start = time.time()
Framework_all.grayscale(TEST_DATA)
end = time.time()
print("All devices: {0}".format(str((end - start) * 1000)), )

start = time.time()
Framework_only_CPU.grayscale(TEST_DATA)
end = time.time()
print("CPU: {0}".format(str((end - start) * 1000)), )

# start = time.time()
# Framework_only_GPU.grayscale(TEST_DATA)
# end = time.time()
# print("GPU: {0}".format(str((end - start) * 1000)), )

# start = time.time()
# Framework_only_iGPU.grayscale(TEST_DATA)
# end = time.time()
# print("iGPU: {0}".format(str((end - start) * 1000)), )

# devices_performances = {'device1': 0.900, 'device2': 1.000, 'device3': 0.900}

# def get_schedule_prio_unused(performances, times):
#     local_performances = performances.copy()
#     fastest_device = max(local_performances, key=local_performances.get)
#     for _ in range(times):
#         current_device = max(local_performances, key=local_performances.get)
#         if current_device is not fastest_device and local_performances[
#                 current_device] > local_performances[fastest_device]:

#             #### execute
#             print(current_device)
#             #### execute
#             local_performances[current_device] = performances[current_device]
#             for device in local_performances:
#                 if device is not current_device:
#                     local_performances[device] = round(
#                         local_performances[device] + performances[device], 3)
#         else:
#             #### execute
#             print(fastest_device)
#             local_performances[fastest_device] = performances[fastest_device]
#             #### execute
#             for device in local_performances:
#                 if device != fastest_device:
#                     local_performances[device] = round(
#                         local_performances[device] + performances[device], 3)

# get_schedule_prio_unused(devices_performances, 30)
