import PyroParallel as Pyro
import numpy as np
import time
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
### TESTING PANNEL
NO_TESTING = -1
TEST_ALL = 0
TEST_ONLY_CPU = 1
TEST_ONLY_GPUS = 2
TEST_ONLY_iGPU = 3
TEST_ONLY_iGPU_CPU = 4

PROCESSING_MODE = Pyro.DEVICE_MODE
TEST_MODE = TEST_ONLY_GPUS

### TESTING PANNEL

Framework_all = Pyro.PyroParallel(verbose=True,
                                  exclude_FPGA=True,
                                  exclude_others=True,
                                  emulation=False,
                                  processing_mode=PROCESSING_MODE)
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

Framework_iGPU_CPU = Pyro.PyroParallel(verbose=True,
                                       exclude_FPGA=True,
                                       exclude_others=True,
                                       emulation=False,
                                       processing_mode=PROCESSING_MODE)
Framework_iGPU_CPU.opencl_devices = Framework_iGPU_CPU.opencl_devices[1:]
Framework_iGPU_CPU.benchmark_api()
Framework_all.benchmark_api()
Framework_only_CPU.benchmark_api()
Framework_only_GPUS.benchmark_api()
Framework_only_iGPU.benchmark_api()

### TEST DATA
NUMBER_OF_TESTS = 120
GRAYSCALE_TEST_DATA = [
    np.full((4000, 4000, 3), 255, dtype=np.uint8)
    for _ in range(NUMBER_OF_TESTS)
]

TEST_FP32_A = [np.full(10000, 1.1648, dtype=np.float32) for _ in range(100)]
TEST_FP32_B = [np.full(10000, 2.2569, dtype=np.float32) for _ in range(100)]

TEST_FP64_A = [np.full(10000, 1.1648, dtype=np.float64) for _ in range(100)]
TEST_FP64_B = [np.full(10000, 2.2569, dtype=np.float64) for _ in range(100)]

input_image = np.array(Image.open("./sample_data/5120x2880_1_NO_HDR.bmp"))
EDGE_DETECTION_SAMPLE_IMAGE = Framework_all.grayscale(
    [input_image for x in range(120)])
### TEST DATA

### FUNCTIONALITY TESTING
if TEST_MODE == NO_TESTING:
    Framework_all.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE, 110)

for _ in range(2):
    PROCESSING_MODE = _

    for _ in range(5):
        TEST_MODE = _
        if TEST_MODE == TEST_ALL:
            processing_mode = ""
            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                processing_mode = "_platform"
            else:
                processing_mode = "_devices"
            file_path = f".\output_test\\all{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_adition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_adition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):

                # grayscale tests
                start = time.time()
                Framework_all.grayscale(GRAYSCALE_TEST_DATA)
                end = time.time()
                timetable_grayscale.append((end - start) * 1000)
                time.sleep(2)

                # fp32 tests
                start = time.time()
                Framework_all.operation(TEST_FP32_A, TEST_FP32_B,
                                        Pyro.ADDITION, Pyro.FP32)
                end = time.time()
                timetable_operation_adition_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_all.operation(TEST_FP32_A, TEST_FP32_B,
                                        Pyro.SUBTRACT, Pyro.FP32)
                end = time.time()
                timetable_operation_subtraction_fp32.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_all.operation(TEST_FP32_A, TEST_FP32_B,
                                        Pyro.MULTIPLY, Pyro.FP32)
                end = time.time()
                timetable_operation_multiply_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_all.operation(TEST_FP32_A, TEST_FP32_B,
                                        Pyro.DIVISION, Pyro.FP32)
                end = time.time()
                timetable_operation_division_fp32.append((end - start) * 1000)
                time.sleep(2)

                # fp64 tests
                start = time.time()
                Framework_all.operation(TEST_FP64_A, TEST_FP64_B,
                                        Pyro.ADDITION, Pyro.FP64)
                end = time.time()
                timetable_operation_adition_fp64.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_all.operation(TEST_FP64_A, TEST_FP64_B,
                                        Pyro.SUBTRACT, Pyro.FP64)
                end = time.time()
                timetable_operation_subtraction_fp64.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_all.operation(TEST_FP64_A, TEST_FP64_B,
                                        Pyro.MULTIPLY, Pyro.FP64)
                end = time.time()
                timetable_operation_multiply_fp64.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_all.operation(TEST_FP64_A, TEST_FP64_B,
                                        Pyro.DIVISION, Pyro.FP64)
                end = time.time()
                timetable_operation_division_fp64.append((end - start) * 1000)
                time.sleep(2)

                # edge detection tests
                start = time.time()
                Framework_all.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE, 256)
                end = time.time()
                timetable_edgedetection.append((end - start) * 1000)
                time.sleep(2)

            avg_grayscale = str(np.average(timetable_grayscale))
            avg_fp32_division = str(
                np.average(timetable_operation_division_fp32))
            avg_fp32_subtraction = str(
                np.average(timetable_operation_subtraction_fp32))
            avg_fp32_adition = str(
                np.average(timetable_operation_adition_fp32))
            avg_fp32_multiply = str(
                np.average(timetable_operation_multiply_fp32))
            avg_fp64_division = str(
                np.average(timetable_operation_division_fp64))
            avg_fp64_subtraction = str(
                np.average(timetable_operation_subtraction_fp64))
            avg_fp64_adition = str(
                np.average(timetable_operation_adition_fp64))
            avg_fp64_multiply = str(
                np.average(timetable_operation_multiply_fp64))
            avg_edgedetection = str(np.average(timetable_edgedetection))

            max_grayscale = str(max(timetable_grayscale))
            max_fp32_multiply = str(max(timetable_operation_multiply_fp32))
            max_fp32_adition = str(max(timetable_operation_adition_fp32))
            max_fp32_subtraction = str(
                max(timetable_operation_subtraction_fp32))
            max_fp32_division = str(max(timetable_operation_division_fp32))
            max_fp64_division = str(max(timetable_operation_division_fp64))
            max_fp64_multiply = str(max(timetable_operation_multiply_fp64))
            max_fp64_adition = str(max(timetable_operation_adition_fp64))
            max_fp64_subtraction = str(
                max(timetable_operation_subtraction_fp64))
            max_edgedetection = str(max(timetable_edgedetection))

            min_grayscale = str(min(timetable_grayscale))
            min_fp32_multiply = str(min(timetable_operation_multiply_fp32))
            min_fp32_adition = str(min(timetable_operation_adition_fp32))
            min_fp32_subtraction = str(
                min(timetable_operation_subtraction_fp32))
            min_fp32_division = str(min(timetable_operation_division_fp32))
            min_fp64_division = str(min(timetable_operation_division_fp64))
            min_fp64_multiply = str(min(timetable_operation_multiply_fp64))
            min_fp64_adition = str(min(timetable_operation_adition_fp64))
            min_fp64_subtraction = str(
                min(timetable_operation_subtraction_fp64))
            min_edgedetection = str(min(timetable_edgedetection))

            res = f"""
grayscale           avg: {avg_grayscale} max: {max_grayscale} min: {min_grayscale} 
fp32_division       avg: {avg_fp32_division} max: {max_fp32_division} min: {min_fp32_division} 
fp32_multiply       avg: {avg_fp32_multiply} max: {max_fp32_multiply} min: {min_fp32_multiply} 
fp32_adition        avg: {avg_fp32_adition} max: {max_fp32_adition} min: {min_fp32_adition} 
fp32_subtraction    avg: {avg_fp32_subtraction} max: {max_fp32_subtraction} min: {min_fp32_subtraction} 
fp64_division       avg: {avg_fp64_division} max: {max_fp64_division} min: {min_fp64_division} 
fp64_multiply       avg: {avg_fp64_multiply} max: {max_fp64_multiply} min: {min_fp64_multiply} 
fp64_adition        avg: {avg_fp64_adition} max: {max_fp64_adition} min: {min_fp64_adition} 
fp64_subtraction    avg: {avg_fp64_subtraction} max: {max_fp64_subtraction} min: {min_fp64_subtraction} 
edge_detection      avg: {avg_edgedetection} max: {max_edgedetection} min: {min_edgedetection} \n\n
            """
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_CPU:
            processing_mode = ""
            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                processing_mode = "_platform"
            else:
                processing_mode = "_devices"
            file_path = f".\output_test\\CPU{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_adition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_adition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):

                # grayscale tests
                start = time.time()
                Framework_only_CPU.grayscale(GRAYSCALE_TEST_DATA)
                end = time.time()
                timetable_grayscale.append((end - start) * 1000)
                time.sleep(2)

                # fp32 tests
                start = time.time()
                Framework_only_CPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.ADDITION, Pyro.FP32)
                end = time.time()
                timetable_operation_adition_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_CPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.SUBTRACT, Pyro.FP32)
                end = time.time()
                timetable_operation_subtraction_fp32.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_CPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.MULTIPLY, Pyro.FP32)
                end = time.time()
                timetable_operation_multiply_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_CPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.DIVISION, Pyro.FP32)
                end = time.time()
                timetable_operation_division_fp32.append((end - start) * 1000)
                time.sleep(2)

                # fp64 tests
                start = time.time()
                Framework_only_CPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.ADDITION, Pyro.FP64)
                end = time.time()
                timetable_operation_adition_fp64.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_CPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.SUBTRACT, Pyro.FP64)
                end = time.time()
                timetable_operation_subtraction_fp64.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_CPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.MULTIPLY, Pyro.FP64)
                end = time.time()
                timetable_operation_multiply_fp64.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_CPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.DIVISION, Pyro.FP64)
                end = time.time()
                timetable_operation_division_fp64.append((end - start) * 1000)
                time.sleep(2)

                # edge detection tests
                start = time.time()
                Framework_only_CPU.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
                                                  256)
                end = time.time()
                timetable_edgedetection.append((end - start) * 1000)
                time.sleep(2)

            avg_grayscale = str(np.average(timetable_grayscale))
            avg_fp32_division = str(
                np.average(timetable_operation_division_fp32))
            avg_fp32_subtraction = str(
                np.average(timetable_operation_subtraction_fp32))
            avg_fp32_adition = str(
                np.average(timetable_operation_adition_fp32))
            avg_fp32_multiply = str(
                np.average(timetable_operation_multiply_fp32))
            avg_fp64_division = str(
                np.average(timetable_operation_division_fp64))
            avg_fp64_subtraction = str(
                np.average(timetable_operation_subtraction_fp64))
            avg_fp64_adition = str(
                np.average(timetable_operation_adition_fp64))
            avg_fp64_multiply = str(
                np.average(timetable_operation_multiply_fp64))
            avg_edgedetection = str(np.average(timetable_edgedetection))

            max_grayscale = str(max(timetable_grayscale))
            max_fp32_multiply = str(max(timetable_operation_multiply_fp32))
            max_fp32_adition = str(max(timetable_operation_adition_fp32))
            max_fp32_subtraction = str(
                max(timetable_operation_subtraction_fp32))
            max_fp32_division = str(max(timetable_operation_division_fp32))
            max_fp64_division = str(max(timetable_operation_division_fp64))
            max_fp64_multiply = str(max(timetable_operation_multiply_fp64))
            max_fp64_adition = str(max(timetable_operation_adition_fp64))
            max_fp64_subtraction = str(
                max(timetable_operation_subtraction_fp64))
            max_edgedetection = str(max(timetable_edgedetection))

            min_grayscale = str(min(timetable_grayscale))
            min_fp32_multiply = str(min(timetable_operation_multiply_fp32))
            min_fp32_adition = str(min(timetable_operation_adition_fp32))
            min_fp32_subtraction = str(
                min(timetable_operation_subtraction_fp32))
            min_fp32_division = str(min(timetable_operation_division_fp32))
            min_fp64_division = str(min(timetable_operation_division_fp64))
            min_fp64_multiply = str(min(timetable_operation_multiply_fp64))
            min_fp64_adition = str(min(timetable_operation_adition_fp64))
            min_fp64_subtraction = str(
                min(timetable_operation_subtraction_fp64))
            min_edgedetection = str(min(timetable_edgedetection))

            res = f"""
grayscale           avg: {avg_grayscale} max: {max_grayscale} min: {min_grayscale} 
fp32_division       avg: {avg_fp32_division} max: {max_fp32_division} min: {min_fp32_division} 
fp32_multiply       avg: {avg_fp32_multiply} max: {max_fp32_multiply} min: {min_fp32_multiply} 
fp32_adition        avg: {avg_fp32_adition} max: {max_fp32_adition} min: {min_fp32_adition} 
fp32_subtraction    avg: {avg_fp32_subtraction} max: {max_fp32_subtraction} min: {min_fp32_subtraction} 
fp64_division       avg: {avg_fp64_division} max: {max_fp64_division} min: {min_fp64_division} 
fp64_multiply       avg: {avg_fp64_multiply} max: {max_fp64_multiply} min: {min_fp64_multiply} 
fp64_adition        avg: {avg_fp64_adition} max: {max_fp64_adition} min: {min_fp64_adition} 
fp64_subtraction    avg: {avg_fp64_subtraction} max: {max_fp64_subtraction} min: {min_fp64_subtraction} 
edge_detection      avg: {avg_edgedetection} max: {max_edgedetection} min: {min_edgedetection} \n\n
            """
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_GPUS:
            processing_mode = ""
            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                processing_mode = "_platform"
            else:
                processing_mode = "_devices"
            file_path = f".\output_test\\GPUS{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_adition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_adition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):

                # grayscale tests
                start = time.time()
                Framework_only_GPUS.grayscale(GRAYSCALE_TEST_DATA)
                end = time.time()
                timetable_grayscale.append((end - start) * 1000)
                time.sleep(2)

                # fp32 tests
                start = time.time()
                Framework_only_GPUS.operation(TEST_FP32_A, TEST_FP32_B,
                                              Pyro.ADDITION, Pyro.FP32)
                end = time.time()
                timetable_operation_adition_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPUS.operation(TEST_FP32_A, TEST_FP32_B,
                                              Pyro.SUBTRACT, Pyro.FP32)
                end = time.time()
                timetable_operation_subtraction_fp32.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPUS.operation(TEST_FP32_A, TEST_FP32_B,
                                              Pyro.MULTIPLY, Pyro.FP32)
                end = time.time()
                timetable_operation_multiply_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPUS.operation(TEST_FP32_A, TEST_FP32_B,
                                              Pyro.DIVISION, Pyro.FP32)
                end = time.time()
                timetable_operation_division_fp32.append((end - start) * 1000)
                time.sleep(2)

                # fp64 tests
                start = time.time()
                Framework_only_GPUS.operation(TEST_FP64_A, TEST_FP64_B,
                                              Pyro.ADDITION, Pyro.FP64)
                end = time.time()
                timetable_operation_adition_fp64.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPUS.operation(TEST_FP64_A, TEST_FP64_B,
                                              Pyro.SUBTRACT, Pyro.FP64)
                end = time.time()
                timetable_operation_subtraction_fp64.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPUS.operation(TEST_FP64_A, TEST_FP64_B,
                                              Pyro.MULTIPLY, Pyro.FP64)
                end = time.time()
                timetable_operation_multiply_fp64.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPUS.operation(TEST_FP64_A, TEST_FP64_B,
                                              Pyro.DIVISION, Pyro.FP64)
                end = time.time()
                timetable_operation_division_fp64.append((end - start) * 1000)
                time.sleep(2)

                # edge detection tests
                start = time.time()
                Framework_only_GPUS.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
                                                   256)
                end = time.time()
                timetable_edgedetection.append((end - start) * 1000)
                time.sleep(2)

            avg_grayscale = str(np.average(timetable_grayscale))
            avg_fp32_division = str(
                np.average(timetable_operation_division_fp32))
            avg_fp32_subtraction = str(
                np.average(timetable_operation_subtraction_fp32))
            avg_fp32_adition = str(
                np.average(timetable_operation_adition_fp32))
            avg_fp32_multiply = str(
                np.average(timetable_operation_multiply_fp32))
            avg_fp64_division = str(
                np.average(timetable_operation_division_fp64))
            avg_fp64_subtraction = str(
                np.average(timetable_operation_subtraction_fp64))
            avg_fp64_adition = str(
                np.average(timetable_operation_adition_fp64))
            avg_fp64_multiply = str(
                np.average(timetable_operation_multiply_fp64))
            avg_edgedetection = str(np.average(timetable_edgedetection))

            max_grayscale = str(max(timetable_grayscale))
            max_fp32_multiply = str(max(timetable_operation_multiply_fp32))
            max_fp32_adition = str(max(timetable_operation_adition_fp32))
            max_fp32_subtraction = str(
                max(timetable_operation_subtraction_fp32))
            max_fp32_division = str(max(timetable_operation_division_fp32))
            max_fp64_division = str(max(timetable_operation_division_fp64))
            max_fp64_multiply = str(max(timetable_operation_multiply_fp64))
            max_fp64_adition = str(max(timetable_operation_adition_fp64))
            max_fp64_subtraction = str(
                max(timetable_operation_subtraction_fp64))
            max_edgedetection = str(max(timetable_edgedetection))

            min_grayscale = str(min(timetable_grayscale))
            min_fp32_multiply = str(min(timetable_operation_multiply_fp32))
            min_fp32_adition = str(min(timetable_operation_adition_fp32))
            min_fp32_subtraction = str(
                min(timetable_operation_subtraction_fp32))
            min_fp32_division = str(min(timetable_operation_division_fp32))
            min_fp64_division = str(min(timetable_operation_division_fp64))
            min_fp64_multiply = str(min(timetable_operation_multiply_fp64))
            min_fp64_adition = str(min(timetable_operation_adition_fp64))
            min_fp64_subtraction = str(
                min(timetable_operation_subtraction_fp64))
            min_edgedetection = str(min(timetable_edgedetection))

            res = f"""
grayscale           avg: {avg_grayscale} max: {max_grayscale} min: {min_grayscale} 
fp32_division       avg: {avg_fp32_division} max: {max_fp32_division} min: {min_fp32_division} 
fp32_multiply       avg: {avg_fp32_multiply} max: {max_fp32_multiply} min: {min_fp32_multiply} 
fp32_adition        avg: {avg_fp32_adition} max: {max_fp32_adition} min: {min_fp32_adition} 
fp32_subtraction    avg: {avg_fp32_subtraction} max: {max_fp32_subtraction} min: {min_fp32_subtraction} 
fp64_division       avg: {avg_fp64_division} max: {max_fp64_division} min: {min_fp64_division} 
fp64_multiply       avg: {avg_fp64_multiply} max: {max_fp64_multiply} min: {min_fp64_multiply} 
fp64_adition        avg: {avg_fp64_adition} max: {max_fp64_adition} min: {min_fp64_adition} 
fp64_subtraction    avg: {avg_fp64_subtraction} max: {max_fp64_subtraction} min: {min_fp64_subtraction} 
edge_detection      avg: {avg_edgedetection} max: {max_edgedetection} min: {min_edgedetection} \n\n
            """
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_iGPU:
            processing_mode = ""
            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                processing_mode = "_platform"
            else:
                processing_mode = "_devices"
            file_path = f".\output_test\\iGPU{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_adition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_adition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):

                # grayscale tests
                start = time.time()
                Framework_only_iGPU.grayscale(GRAYSCALE_TEST_DATA)
                end = time.time()
                timetable_grayscale.append((end - start) * 1000)
                time.sleep(2)

                # fp32 tests
                start = time.time()
                Framework_only_iGPU.operation(TEST_FP32_A, TEST_FP32_B,
                                              Pyro.ADDITION, Pyro.FP32)
                end = time.time()
                timetable_operation_adition_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_iGPU.operation(TEST_FP32_A, TEST_FP32_B,
                                              Pyro.SUBTRACT, Pyro.FP32)
                end = time.time()
                timetable_operation_subtraction_fp32.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_iGPU.operation(TEST_FP32_A, TEST_FP32_B,
                                              Pyro.MULTIPLY, Pyro.FP32)
                end = time.time()
                timetable_operation_multiply_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_iGPU.operation(TEST_FP32_A, TEST_FP32_B,
                                              Pyro.DIVISION, Pyro.FP32)
                end = time.time()
                timetable_operation_division_fp32.append((end - start) * 1000)
                time.sleep(2)

                # edge detection tests
                start = time.time()
                Framework_only_iGPU.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
                                                   256)
                end = time.time()
                timetable_edgedetection.append((end - start) * 1000)
                time.sleep(2)

            avg_grayscale = str(np.average(timetable_grayscale))
            avg_fp32_division = str(
                np.average(timetable_operation_division_fp32))
            avg_fp32_subtraction = str(
                np.average(timetable_operation_subtraction_fp32))
            avg_fp32_adition = str(
                np.average(timetable_operation_adition_fp32))
            avg_fp32_multiply = str(
                np.average(timetable_operation_multiply_fp32))
            avg_fp64_division = "N/A"
            avg_fp64_subtraction = "N/A"
            avg_fp64_adition = "N/A"
            avg_fp64_multiply = "N/A"
            avg_edgedetection = str(np.average(timetable_edgedetection))

            max_grayscale = str(max(timetable_grayscale))
            max_fp32_multiply = str(max(timetable_operation_multiply_fp32))
            max_fp32_adition = str(max(timetable_operation_adition_fp32))
            max_fp32_subtraction = str(
                max(timetable_operation_subtraction_fp32))
            max_fp32_division = str(max(timetable_operation_division_fp32))
            max_fp64_division = "N/A"
            max_fp64_multiply = "N/A"
            max_fp64_adition = "N/A"
            max_fp64_subtraction = "N/A"
            max_edgedetection = str(max(timetable_edgedetection))

            min_grayscale = str(min(timetable_grayscale))
            min_fp32_multiply = str(min(timetable_operation_multiply_fp32))
            min_fp32_adition = str(min(timetable_operation_adition_fp32))
            min_fp32_subtraction = str(
                min(timetable_operation_subtraction_fp32))
            min_fp32_division = str(min(timetable_operation_division_fp32))
            min_fp64_division = "N/A"
            min_fp64_multiply = "N/A"
            min_fp64_adition = "N/A"
            min_fp64_subtraction = "N/A"
            min_edgedetection = str(min(timetable_edgedetection))

            res = f"""
grayscale           avg: {avg_grayscale} max: {max_grayscale} min: {min_grayscale} 
fp32_division       avg: {avg_fp32_division} max: {max_fp32_division} min: {min_fp32_division} 
fp32_multiply       avg: {avg_fp32_multiply} max: {max_fp32_multiply} min: {min_fp32_multiply} 
fp32_adition        avg: {avg_fp32_adition} max: {max_fp32_adition} min: {min_fp32_adition} 
fp32_subtraction    avg: {avg_fp32_subtraction} max: {max_fp32_subtraction} min: {min_fp32_subtraction} 
fp64_division       avg: {avg_fp64_division} max: {max_fp64_division} min: {min_fp64_division} 
fp64_multiply       avg: {avg_fp64_multiply} max: {max_fp64_multiply} min: {min_fp64_multiply} 
fp64_adition        avg: {avg_fp64_adition} max: {max_fp64_adition} min: {min_fp64_adition} 
fp64_subtraction    avg: {avg_fp64_subtraction} max: {max_fp64_subtraction} min: {min_fp64_subtraction} 
edge_detection      avg: {avg_edgedetection} max: {max_edgedetection} min: {min_edgedetection} \n\n
            """
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_iGPU_CPU:
            processing_mode = ""
            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                processing_mode = "_platform"
            else:
                processing_mode = "_devices"
            file_path = f".\output_test\\iGPU_CPU{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_adition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_adition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):

                # grayscale tests
                start = time.time()
                Framework_iGPU_CPU.grayscale(GRAYSCALE_TEST_DATA)
                end = time.time()
                timetable_grayscale.append((end - start) * 1000)
                time.sleep(2)

                # fp32 tests
                start = time.time()
                Framework_iGPU_CPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.ADDITION, Pyro.FP32)
                end = time.time()
                timetable_operation_adition_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_iGPU_CPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.SUBTRACT, Pyro.FP32)
                end = time.time()
                timetable_operation_subtraction_fp32.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_iGPU_CPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.MULTIPLY, Pyro.FP32)
                end = time.time()
                timetable_operation_multiply_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_iGPU_CPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.DIVISION, Pyro.FP32)
                end = time.time()
                timetable_operation_division_fp32.append((end - start) * 1000)
                time.sleep(2)

                # fp64 tests

                start = time.time()
                Framework_iGPU_CPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.ADDITION, Pyro.FP64)
                end = time.time()
                timetable_operation_adition_fp64.append((end - start) * 1000)

                time.sleep(2)

                start = time.time()
                Framework_iGPU_CPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.SUBTRACT, Pyro.FP64)
                end = time.time()
                timetable_operation_subtraction_fp64.append(
                    (end - start) * 1000)

                time.sleep(2)

                start = time.time()
                Framework_iGPU_CPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.MULTIPLY, Pyro.FP64)
                end = time.time()
                timetable_operation_multiply_fp64.append((end - start) * 1000)

                time.sleep(2)

                start = time.time()
                Framework_iGPU_CPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.DIVISION, Pyro.FP64)
                end = time.time()
                timetable_operation_division_fp64.append((end - start) * 1000)

                time.sleep(2)

                # edge detection tests
                start = time.time()
                Framework_iGPU_CPU.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
                                                  256)
                end = time.time()
                timetable_edgedetection.append((end - start) * 1000)
                time.sleep(2)

            avg_grayscale = str(np.average(timetable_grayscale))
            avg_fp32_division = str(
                np.average(timetable_operation_division_fp32))
            avg_fp32_subtraction = str(
                np.average(timetable_operation_subtraction_fp32))
            avg_fp32_adition = str(
                np.average(timetable_operation_adition_fp32))
            avg_fp32_multiply = str(
                np.average(timetable_operation_multiply_fp32))
            avg_fp64_division = str(
                np.average(timetable_operation_division_fp64))
            avg_fp64_subtraction = str(
                np.average(timetable_operation_subtraction_fp64))
            avg_fp64_adition = str(
                np.average(timetable_operation_adition_fp64))
            avg_fp64_multiply = str(
                np.average(timetable_operation_multiply_fp64))
            avg_edgedetection = str(np.average(timetable_edgedetection))

            max_grayscale = str(max(timetable_grayscale))
            max_fp32_multiply = str(max(timetable_operation_multiply_fp32))
            max_fp32_adition = str(max(timetable_operation_adition_fp32))
            max_fp32_subtraction = str(
                max(timetable_operation_subtraction_fp32))
            max_fp32_division = str(max(timetable_operation_division_fp32))
            max_fp64_division = str(max(timetable_operation_division_fp64))
            max_fp64_multiply = str(max(timetable_operation_multiply_fp64))
            max_fp64_adition = str(max(timetable_operation_adition_fp64))
            max_fp64_subtraction = str(
                max(timetable_operation_subtraction_fp64))
            max_edgedetection = str(max(timetable_edgedetection))
            avg_fp64_division = str(
                np.average(timetable_operation_division_fp64))
            avg_fp64_subtraction = str(
                np.average(timetable_operation_subtraction_fp64))
            avg_fp64_adition = str(
                np.average(timetable_operation_adition_fp64))
            avg_fp64_multiply = str(
                np.average(timetable_operation_multiply_fp64))

            min_grayscale = str(min(timetable_grayscale))
            min_fp32_multiply = str(min(timetable_operation_multiply_fp32))
            min_fp32_adition = str(min(timetable_operation_adition_fp32))
            min_fp32_subtraction = str(
                min(timetable_operation_subtraction_fp32))
            min_fp32_division = str(min(timetable_operation_division_fp32))
            min_fp64_division = str(min(timetable_operation_division_fp64))
            min_fp64_multiply = str(min(timetable_operation_multiply_fp64))
            min_fp64_adition = str(min(timetable_operation_adition_fp64))
            min_fp64_subtraction = str(
                min(timetable_operation_subtraction_fp64))
            min_edgedetection = str(min(timetable_edgedetection))

            res = f"""
grayscale           avg: {avg_grayscale} max: {max_grayscale} min: {min_grayscale} 
fp32_division       avg: {avg_fp32_division} max: {max_fp32_division} min: {min_fp32_division} 
fp32_multiply       avg: {avg_fp32_multiply} max: {max_fp32_multiply} min: {min_fp32_multiply} 
fp32_adition        avg: {avg_fp32_adition} max: {max_fp32_adition} min: {min_fp32_adition} 
fp32_subtraction    avg: {avg_fp32_subtraction} max: {max_fp32_subtraction} min: {min_fp32_subtraction} 
fp64_division       avg: {avg_fp64_division} max: {max_fp64_division} min: {min_fp64_division} 
fp64_multiply       avg: {avg_fp64_multiply} max: {max_fp64_multiply} min: {min_fp64_multiply} 
fp64_adition        avg: {avg_fp64_adition} max: {max_fp64_adition} min: {min_fp64_adition} 
fp64_subtraction    avg: {avg_fp64_subtraction} max: {max_fp64_subtraction} min: {min_fp64_subtraction} 
edge_detection      avg: {avg_edgedetection} max: {max_edgedetection} min: {min_edgedetection} \n\n
            """
            file_results.write(res)
            file_results.close()
print("done..")
