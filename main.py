from base.PyroParallelApi import PyroParallel
from base.hardware_resources.OpenCLVersion import *
if __name__ == "__main__":
    pp = PyroParallel(verbose=True)
    # x = 2
    # total_time = 0.1
    # for i in range(10):
    #     x = 2
    #     start_time = time.time()
    #     for i in range(1024):
    #         x = x * 2
    #     end_time = time.time()
    #     total_time += (end_time - start_time)
    # execution_time = total_time / 10.0
    # print("Execution time:", execution_time, "seconds")
    x = 10
