import PyroParallel as PyroParallel

Framework = PyroParallel.PyroParallel(verbose=True,
                                      exclude_FPGA=True,
                                      exclude_others=True,
                                      emulation=False)
# Framework2 = PyroParallel.PyroParallel(verbose=True,
#                                        emulation=True,
#                                        empty_platform=True)
#                                        exclude_FPGA=False,
#                                        exclude_others=False,
#                                        emulation=True)

# Framework.opencl_devices[0] = Framework.opencl_devices[1]
# Framework.opencl_devices[2] = Framework.opencl_devices[1]
# z = Framework.get_all_devices_contexts()
# y = Framework.get_all_platform_contexts()
Framework.benchmark_api()
x = 12
# total_time = 0.1
# # for i in range(10):
# #     x = 2
# #     start_time = time.time()
# #     for i in range(1024):
# #         x = x * 2
# #     end_time = time.time()
# #     total_time += (end_time - start_time)
# # execution_time = total_time / 10.0
# # print("Execution time:", execution_time, "seconds")
# x = 10
