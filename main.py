import PyroParallel as Pyro
import numpy as np
import time
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
### TESTING PANNEL
NO_TESTING = -1
TEST_ALL = 0
TEST_ONLY_CPU = 1
TEST_ONLY_GPUS = 2
TEST_ONLY_iGPU = 3
TEST_ONLY_iGPU_CPU = 4
TEST_ONLY_GPU = 5

PROCESSING_MODE = Pyro.DEVICE_MODE
TEST_MODE = -1
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
Framework_only_GPU = Pyro.PyroParallel(verbose=True,
                                       exclude_FPGA=True,
                                       exclude_others=True,
                                       emulation=False,
                                       processing_mode=PROCESSING_MODE,
                                       exclude_CPU=True)
Framework_only_GPU.opencl_devices = Framework_only_GPU.opencl_devices[0:1]

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
Framework_only_GPU.benchmark_api()
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
    print(f"platform: {str(_)}")
    PROCESSING_MODE = _
    if _ == Pyro.DEVICE_MODE:
        processing_mode = "_device"
    elif _ == Pyro.PLATFORM_MODE:
        processing_mode = "_platform"
    for _ in range(6):
        print(f"device:{str(_)}")
        TEST_MODE = _
        if TEST_MODE == TEST_ALL:

            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                Framework_all.set_processing_platform_mode()
            else:
                Framework_all.set_processing_device_mode()
            file_path = f".\output_test\\all{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_addition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_addition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):
                print(f"test:{str(_)}")
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
                timetable_operation_addition_fp32.append((end - start) * 1000)
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
                timetable_operation_addition_fp64.append((end - start) * 1000)
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

            avg_grayscale = str(round(np.average(timetable_grayscale), 3))
            avg_fp32_division = str(
                round(np.average(timetable_operation_division_fp32), 3))
            avg_fp32_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp32), 3))
            avg_fp32_addition = str(
                round(np.average(timetable_operation_addition_fp32), 3))
            avg_fp32_multiply = str(
                round(np.average(timetable_operation_multiply_fp32), 3))
            avg_fp64_division = str(
                round(np.average(timetable_operation_division_fp64), 3))
            avg_fp64_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp64), 3))
            avg_fp64_addition = str(
                round(np.average(timetable_operation_addition_fp64), 3))
            avg_fp64_multiply = str(
                round(np.average(timetable_operation_multiply_fp64), 3))
            avg_edgedetection = str(
                round(np.average(timetable_edgedetection), 3))

            max_grayscale = str(round(max(timetable_grayscale), 3))
            max_fp32_multiply = str(
                round(max(timetable_operation_multiply_fp32), 3))
            max_fp32_addition = str(
                round(max(timetable_operation_addition_fp32), 3))
            max_fp32_subtraction = str(
                round(max(timetable_operation_subtraction_fp32), 3))
            max_fp32_division = str(
                round(max(timetable_operation_division_fp32), 3))
            max_fp64_division = str(
                round(max(timetable_operation_division_fp64), 3))
            max_fp64_multiply = str(
                round(max(timetable_operation_multiply_fp64), 3))
            max_fp64_addition = str(
                round(max(timetable_operation_addition_fp64), 3))
            max_fp64_subtraction = str(
                round(max(timetable_operation_subtraction_fp64), 3))
            max_edgedetection = str(round(max(timetable_edgedetection), 3))

            min_grayscale = str(round(min(timetable_grayscale), 3))
            min_fp32_multiply = str(
                round(min(timetable_operation_multiply_fp32), 3))
            min_fp32_addition = str(
                round(min(timetable_operation_addition_fp32), 3))
            min_fp32_subtraction = str(
                round(min(timetable_operation_subtraction_fp32), 3))
            min_fp32_division = str(
                round(min(timetable_operation_division_fp32), 3))
            min_fp64_division = str(
                round(min(timetable_operation_division_fp64), 3))
            min_fp64_multiply = str(
                round(min(timetable_operation_multiply_fp64), 3))
            min_fp64_addition = str(
                round(min(timetable_operation_addition_fp64), 3))
            min_fp64_subtraction = str(
                round(min(timetable_operation_subtraction_fp64), 3))
            min_edgedetection = str(round(min(timetable_edgedetection), 3))

            res = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Measurements}}
\\label{{tab:performance}}
\\begin{{tabular}}{{|l|l|l|l|}}
\\hline
grayscale & avg: {avg_grayscale} & max: {max_grayscale} & min: {min_grayscale} \\\\
fp32\_division & avg: {avg_fp32_division} & max: {max_fp32_division} & min: {min_fp32_division} \\\\
fp32\_multiply & avg: {avg_fp32_multiply} & max: {max_fp32_multiply} & min: {min_fp32_multiply} \\\\
fp32\_addition & avg: {avg_fp32_addition} & max: {max_fp32_addition} & min: {min_fp32_addition} \\\\
fp32\_subtraction & avg: {avg_fp32_subtraction} & max: {max_fp32_subtraction} & min: {min_fp32_subtraction} \\\\
fp64\_division & avg: {avg_fp64_division} & max: {max_fp64_division} & min: {min_fp64_division} \\\\
fp64\_multiply & avg: {avg_fp64_multiply} & max: {max_fp64_multiply} & min: {min_fp64_multiply} \\\\
fp64\_addition & avg: {avg_fp64_addition} & max: {max_fp64_addition} & min: {min_fp64_addition} \\\\
fp64\_subtraction & avg: {avg_fp64_subtraction} & max: {max_fp64_subtraction} & min: {min_fp64_subtraction} \\\\
edge\_detection & avg: {avg_edgedetection} & max: {max_edgedetection} & min: {min_edgedetection} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_CPU:

            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                Framework_all.set_processing_platform_mode()
            else:
                Framework_all.set_processing_device_mode()
            file_path = f".\output_test\\CPU{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_addition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_addition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):
                print(f"test:{str(_)}")
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
                timetable_operation_addition_fp32.append((end - start) * 1000)
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
                timetable_operation_addition_fp64.append((end - start) * 1000)
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

            avg_grayscale = str(round(np.average(timetable_grayscale), 3))
            avg_fp32_division = str(
                round(np.average(timetable_operation_division_fp32), 3))
            avg_fp32_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp32), 3))
            avg_fp32_addition = str(
                round(np.average(timetable_operation_addition_fp32), 3))
            avg_fp32_multiply = str(
                round(np.average(timetable_operation_multiply_fp32), 3))
            avg_fp64_division = str(
                round(np.average(timetable_operation_division_fp64), 3))
            avg_fp64_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp64), 3))
            avg_fp64_addition = str(
                round(np.average(timetable_operation_addition_fp64), 3))
            avg_fp64_multiply = str(
                round(np.average(timetable_operation_multiply_fp64), 3))
            avg_edgedetection = str(
                round(np.average(timetable_edgedetection), 3))

            max_grayscale = str(round(max(timetable_grayscale), 3))
            max_fp32_multiply = str(
                round(max(timetable_operation_multiply_fp32), 3))
            max_fp32_addition = str(
                round(max(timetable_operation_addition_fp32), 3))
            max_fp32_subtraction = str(
                round(max(timetable_operation_subtraction_fp32), 3))
            max_fp32_division = str(
                round(max(timetable_operation_division_fp32), 3))
            max_fp64_division = str(
                round(max(timetable_operation_division_fp64), 3))
            max_fp64_multiply = str(
                round(max(timetable_operation_multiply_fp64), 3))
            max_fp64_addition = str(
                round(max(timetable_operation_addition_fp64), 3))
            max_fp64_subtraction = str(
                round(max(timetable_operation_subtraction_fp64), 3))
            max_edgedetection = str(round(max(timetable_edgedetection), 3))

            min_grayscale = str(round(min(timetable_grayscale), 3))
            min_fp32_multiply = str(
                round(min(timetable_operation_multiply_fp32), 3))
            min_fp32_addition = str(
                round(min(timetable_operation_addition_fp32), 3))
            min_fp32_subtraction = str(
                round(min(timetable_operation_subtraction_fp32), 3))
            min_fp32_division = str(
                round(min(timetable_operation_division_fp32), 3))
            min_fp64_division = str(
                round(min(timetable_operation_division_fp64), 3))
            min_fp64_multiply = str(
                round(min(timetable_operation_multiply_fp64), 3))
            min_fp64_addition = str(
                round(min(timetable_operation_addition_fp64), 3))
            min_fp64_subtraction = str(
                round(min(timetable_operation_subtraction_fp64), 3))
            min_edgedetection = str(round(min(timetable_edgedetection), 3))

            res = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Measurements}}
\\label{{tab:performance}}
\\begin{{tabular}}{{|l|l|l|l|}}
\\hline
grayscale & avg: {avg_grayscale} & max: {max_grayscale} & min: {min_grayscale} \\\\
fp32\_division & avg: {avg_fp32_division} & max: {max_fp32_division} & min: {min_fp32_division} \\\\
fp32\_multiply & avg: {avg_fp32_multiply} & max: {max_fp32_multiply} & min: {min_fp32_multiply} \\\\
fp32\_addition & avg: {avg_fp32_addition} & max: {max_fp32_addition} & min: {min_fp32_addition} \\\\
fp32\_subtraction & avg: {avg_fp32_subtraction} & max: {max_fp32_subtraction} & min: {min_fp32_subtraction} \\\\
fp64\_division & avg: {avg_fp64_division} & max: {max_fp64_division} & min: {min_fp64_division} \\\\
fp64\_multiply & avg: {avg_fp64_multiply} & max: {max_fp64_multiply} & min: {min_fp64_multiply} \\\\
fp64\_addition & avg: {avg_fp64_addition} & max: {max_fp64_addition} & min: {min_fp64_addition} \\\\
fp64\_subtraction & avg: {avg_fp64_subtraction} & max: {max_fp64_subtraction} & min: {min_fp64_subtraction} \\\\
edge\_detection & avg: {avg_edgedetection} & max: {max_edgedetection} & min: {min_edgedetection} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_GPUS:

            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                Framework_all.set_processing_platform_mode()
            else:
                Framework_all.set_processing_device_mode()
            file_path = f".\output_test\\GPUS{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_addition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_addition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):
                print(f"test:{str(_)}")
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
                timetable_operation_addition_fp32.append((end - start) * 1000)
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
                timetable_operation_addition_fp64.append((end - start) * 1000)
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

            avg_grayscale = str(round(np.average(timetable_grayscale), 3))
            avg_fp32_division = str(
                round(np.average(timetable_operation_division_fp32), 3))
            avg_fp32_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp32), 3))
            avg_fp32_addition = str(
                round(np.average(timetable_operation_addition_fp32), 3))
            avg_fp32_multiply = str(
                round(np.average(timetable_operation_multiply_fp32), 3))
            avg_fp64_division = str(
                round(np.average(timetable_operation_division_fp64), 3))
            avg_fp64_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp64), 3))
            avg_fp64_addition = str(
                round(np.average(timetable_operation_addition_fp64), 3))
            avg_fp64_multiply = str(
                round(np.average(timetable_operation_multiply_fp64), 3))
            avg_edgedetection = str(
                round(np.average(timetable_edgedetection), 3))

            max_grayscale = str(round(max(timetable_grayscale), 3))
            max_fp32_multiply = str(
                round(max(timetable_operation_multiply_fp32), 3))
            max_fp32_addition = str(
                round(max(timetable_operation_addition_fp32), 3))
            max_fp32_subtraction = str(
                round(max(timetable_operation_subtraction_fp32), 3))
            max_fp32_division = str(
                round(max(timetable_operation_division_fp32), 3))
            max_fp64_division = str(
                round(max(timetable_operation_division_fp64), 3))
            max_fp64_multiply = str(
                round(max(timetable_operation_multiply_fp64), 3))
            max_fp64_addition = str(
                round(max(timetable_operation_addition_fp64), 3))
            max_fp64_subtraction = str(
                round(max(timetable_operation_subtraction_fp64), 3))
            max_edgedetection = str(round(max(timetable_edgedetection), 3))

            min_grayscale = str(round(min(timetable_grayscale), 3))
            min_fp32_multiply = str(
                round(min(timetable_operation_multiply_fp32), 3))
            min_fp32_addition = str(
                round(min(timetable_operation_addition_fp32), 3))
            min_fp32_subtraction = str(
                round(min(timetable_operation_subtraction_fp32), 3))
            min_fp32_division = str(
                round(min(timetable_operation_division_fp32), 3))
            min_fp64_division = str(
                round(min(timetable_operation_division_fp64), 3))
            min_fp64_multiply = str(
                round(min(timetable_operation_multiply_fp64), 3))
            min_fp64_addition = str(
                round(min(timetable_operation_addition_fp64), 3))
            min_fp64_subtraction = str(
                round(min(timetable_operation_subtraction_fp64), 3))
            min_edgedetection = str(round(min(timetable_edgedetection), 3))

            res = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Measurements}}
\\label{{tab:performance}}
\\begin{{tabular}}{{|l|l|l|l|}}
\\hline
grayscale & avg: {avg_grayscale} & max: {max_grayscale} & min: {min_grayscale} \\\\
fp32\_division & avg: {avg_fp32_division} & max: {max_fp32_division} & min: {min_fp32_division} \\\\
fp32\_multiply & avg: {avg_fp32_multiply} & max: {max_fp32_multiply} & min: {min_fp32_multiply} \\\\
fp32\_addition & avg: {avg_fp32_addition} & max: {max_fp32_addition} & min: {min_fp32_addition} \\\\
fp32\_subtraction & avg: {avg_fp32_subtraction} & max: {max_fp32_subtraction} & min: {min_fp32_subtraction} \\\\
fp64\_division & avg: {avg_fp64_division} & max: {max_fp64_division} & min: {min_fp64_division} \\\\
fp64\_multiply & avg: {avg_fp64_multiply} & max: {max_fp64_multiply} & min: {min_fp64_multiply} \\\\
fp64\_addition & avg: {avg_fp64_addition} & max: {max_fp64_addition} & min: {min_fp64_addition} \\\\
fp64\_subtraction & avg: {avg_fp64_subtraction} & max: {max_fp64_subtraction} & min: {min_fp64_subtraction} \\\\
edge\_detection & avg: {avg_edgedetection} & max: {max_edgedetection} & min: {min_edgedetection} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_iGPU:

            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                Framework_all.set_processing_platform_mode()
            else:
                Framework_all.set_processing_device_mode()
            file_path = f".\output_test\\iGPU{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_addition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_addition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):
                print(f"test:{str(_)}")
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
                timetable_operation_addition_fp32.append((end - start) * 1000)
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
            avg_grayscale = str(round(np.average(timetable_grayscale), 3))

            avg_fp32_division = str(
                round(np.average(timetable_operation_division_fp32), 3))
            avg_fp32_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp32), 3))
            avg_fp32_addition = str(
                round(np.average(timetable_operation_addition_fp32), 3))
            avg_fp32_multiply = str(
                round(np.average(timetable_operation_multiply_fp32), 3))
            avg_fp64_division = "N/A"
            avg_fp64_subtraction = "N/A"
            avg_fp64_addition = "N/A"
            avg_fp64_multiply = "N/A"
            avg_edgedetection = str(
                round(np.average(timetable_edgedetection), 3))

            max_grayscale = str(round(max(timetable_grayscale), 3))
            max_fp32_multiply = str(
                round(max(timetable_operation_multiply_fp32), 3))
            max_fp32_addition = str(
                round(max(timetable_operation_addition_fp32), 3))
            max_fp32_subtraction = str(
                round(max(timetable_operation_subtraction_fp32), 3))
            max_fp32_division = str(
                round(max(timetable_operation_division_fp32), 3))
            max_fp64_division = "N/A"
            max_fp64_multiply = "N/A"
            max_fp64_addition = "N/A"
            max_fp64_subtraction = "N/A"
            max_edgedetection = str(round(max(timetable_edgedetection), 3))

            min_grayscale = str(round(min(timetable_grayscale), 3))
            min_fp32_multiply = str(
                round(min(timetable_operation_multiply_fp32), 3))
            min_fp32_addition = str(
                round(min(timetable_operation_addition_fp32), 3))
            min_fp32_subtraction = str(
                round(min(timetable_operation_subtraction_fp32), 3))
            min_fp32_division = str(
                round(min(timetable_operation_division_fp32), 3))
            min_fp64_division = "N/A"
            min_fp64_multiply = "N/A"
            min_fp64_addition = "N/A"
            min_fp64_subtraction = "N/A"
            min_edgedetection = str(round(min(timetable_edgedetection), 3))

            res = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Measurements}}
\\label{{tab:performance}}
\\begin{{tabular}}{{|l|l|l|l|}}
\\hline
grayscale & avg: {avg_grayscale} & max: {max_grayscale} & min: {min_grayscale} \\\\
fp32\_division & avg: {avg_fp32_division} & max: {max_fp32_division} & min: {min_fp32_division} \\\\
fp32\_multiply & avg: {avg_fp32_multiply} & max: {max_fp32_multiply} & min: {min_fp32_multiply} \\\\
fp32\_addition & avg: {avg_fp32_addition} & max: {max_fp32_addition} & min: {min_fp32_addition} \\\\
fp32\_subtraction & avg: {avg_fp32_subtraction} & max: {max_fp32_subtraction} & min: {min_fp32_subtraction} \\\\
fp64\_division & avg: {avg_fp64_division} & max: {max_fp64_division} & min: {min_fp64_division} \\\\
fp64\_multiply & avg: {avg_fp64_multiply} & max: {max_fp64_multiply} & min: {min_fp64_multiply} \\\\
fp64\_addition & avg: {avg_fp64_addition} & max: {max_fp64_addition} & min: {min_fp64_addition} \\\\
fp64\_subtraction & avg: {avg_fp64_subtraction} & max: {max_fp64_subtraction} & min: {min_fp64_subtraction} \\\\
edge\_detection & avg: {avg_edgedetection} & max: {max_edgedetection} & min: {min_edgedetection} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_iGPU_CPU:
            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                Framework_all.set_processing_platform_mode()
            else:
                Framework_all.set_processing_device_mode()
            file_path = f".\output_test\\iGPU_CPU{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_addition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_addition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):
                print(f"test:{str(_)}")
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
                timetable_operation_addition_fp32.append((end - start) * 1000)
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
                timetable_operation_addition_fp64.append((end - start) * 1000)

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

            avg_grayscale = str(round(np.average(timetable_grayscale), 3))
            avg_fp32_division = str(
                round(np.average(timetable_operation_division_fp32), 3))
            avg_fp32_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp32), 3))
            avg_fp32_addition = str(
                round(np.average(timetable_operation_addition_fp32), 3))
            avg_fp32_multiply = str(
                round(np.average(timetable_operation_multiply_fp32), 3))
            avg_fp64_division = str(
                round(np.average(timetable_operation_division_fp64), 3))
            avg_fp64_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp64), 3))
            avg_fp64_addition = str(
                round(np.average(timetable_operation_addition_fp64), 3))
            avg_fp64_multiply = str(
                round(np.average(timetable_operation_multiply_fp64), 3))
            avg_edgedetection = str(
                round(np.average(timetable_edgedetection), 3))

            max_grayscale = str(round(max(timetable_grayscale), 3))
            max_fp32_multiply = str(
                round(max(timetable_operation_multiply_fp32), 3))
            max_fp32_addition = str(
                round(max(timetable_operation_addition_fp32), 3))
            max_fp32_subtraction = str(
                round(max(timetable_operation_subtraction_fp32), 3))
            max_fp32_division = str(
                round(max(timetable_operation_division_fp32), 3))
            max_fp64_division = str(
                round(max(timetable_operation_division_fp64), 3))
            max_fp64_multiply = str(
                round(max(timetable_operation_multiply_fp64), 3))
            max_fp64_addition = str(
                round(max(timetable_operation_addition_fp64), 3))
            max_fp64_subtraction = str(
                round(max(timetable_operation_subtraction_fp64), 3))
            max_edgedetection = str(round(max(timetable_edgedetection), 3))

            min_grayscale = str(round(min(timetable_grayscale), 3))
            min_fp32_multiply = str(
                round(min(timetable_operation_multiply_fp32), 3))
            min_fp32_addition = str(
                round(min(timetable_operation_addition_fp32), 3))
            min_fp32_subtraction = str(
                round(min(timetable_operation_subtraction_fp32), 3))
            min_fp32_division = str(
                round(min(timetable_operation_division_fp32), 3))
            min_fp64_division = str(
                round(min(timetable_operation_division_fp64), 3))
            min_fp64_multiply = str(
                round(min(timetable_operation_multiply_fp64), 3))
            min_fp64_addition = str(
                round(min(timetable_operation_addition_fp64), 3))
            min_fp64_subtraction = str(
                round(min(timetable_operation_subtraction_fp64), 3))
            min_edgedetection = str(round(min(timetable_edgedetection), 3))

            res = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Measurements}}
\\label{{tab:performance}}
\\begin{{tabular}}{{|l|l|l|l|}}
\\hline
grayscale & avg: {avg_grayscale} & max: {max_grayscale} & min: {min_grayscale} \\\\
fp32\_division & avg: {avg_fp32_division} & max: {max_fp32_division} & min: {min_fp32_division} \\\\
fp32\_multiply & avg: {avg_fp32_multiply} & max: {max_fp32_multiply} & min: {min_fp32_multiply} \\\\
fp32\_addition & avg: {avg_fp32_addition} & max: {max_fp32_addition} & min: {min_fp32_addition} \\\\
fp32\_subtraction & avg: {avg_fp32_subtraction} & max: {max_fp32_subtraction} & min: {min_fp32_subtraction} \\\\
fp64\_division & avg: {avg_fp64_division} & max: {max_fp64_division} & min: {min_fp64_division} \\\\
fp64\_multiply & avg: {avg_fp64_multiply} & max: {max_fp64_multiply} & min: {min_fp64_multiply} \\\\
fp64\_addition & avg: {avg_fp64_addition} & max: {max_fp64_addition} & min: {min_fp64_addition} \\\\
fp64\_subtraction & avg: {avg_fp64_subtraction} & max: {max_fp64_subtraction} & min: {min_fp64_subtraction} \\\\
edge\_detection & avg: {avg_edgedetection} & max: {max_edgedetection} & min: {min_edgedetection} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            file_results.write(res)
            file_results.close()

        if TEST_MODE == TEST_ONLY_GPU:
            if PROCESSING_MODE == Pyro.PLATFORM_MODE:
                Framework_all.set_processing_platform_mode()
            else:
                Framework_all.set_processing_device_mode()
            file_path = f".\output_test\\GPU_{processing_mode}.txt"
            file_results = open(file_path, "w")
            timetable_grayscale = []
            timetable_operation_addition_fp32 = []
            timetable_operation_division_fp32 = []
            timetable_operation_multiply_fp32 = []
            timetable_operation_subtraction_fp32 = []
            timetable_operation_addition_fp64 = []
            timetable_operation_division_fp64 = []
            timetable_operation_multiply_fp64 = []
            timetable_operation_subtraction_fp64 = []
            timetable_edgedetection = []
            for _ in range(10):
                print(f"test:{str(_)}")
                # grayscale tests
                start = time.time()
                Framework_only_GPU.grayscale(GRAYSCALE_TEST_DATA)
                end = time.time()
                timetable_grayscale.append((end - start) * 1000)
                time.sleep(2)

                # fp32 tests
                start = time.time()
                Framework_only_GPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.ADDITION, Pyro.FP32)
                end = time.time()
                timetable_operation_addition_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.SUBTRACT, Pyro.FP32)
                end = time.time()
                timetable_operation_subtraction_fp32.append(
                    (end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.MULTIPLY, Pyro.FP32)
                end = time.time()
                timetable_operation_multiply_fp32.append((end - start) * 1000)
                time.sleep(2)

                start = time.time()
                Framework_only_GPU.operation(TEST_FP32_A, TEST_FP32_B,
                                             Pyro.DIVISION, Pyro.FP32)
                end = time.time()
                timetable_operation_division_fp32.append((end - start) * 1000)
                time.sleep(2)

                # fp64 tests

                start = time.time()
                Framework_only_GPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.ADDITION, Pyro.FP64)
                end = time.time()
                timetable_operation_addition_fp64.append((end - start) * 1000)

                time.sleep(2)

                start = time.time()
                Framework_only_GPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.SUBTRACT, Pyro.FP64)
                end = time.time()
                timetable_operation_subtraction_fp64.append(
                    (end - start) * 1000)

                time.sleep(2)

                start = time.time()
                Framework_only_GPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.MULTIPLY, Pyro.FP64)
                end = time.time()
                timetable_operation_multiply_fp64.append((end - start) * 1000)

                time.sleep(2)

                start = time.time()
                Framework_only_GPU.operation(TEST_FP64_A, TEST_FP64_B,
                                             Pyro.DIVISION, Pyro.FP64)
                end = time.time()
                timetable_operation_division_fp64.append((end - start) * 1000)

                time.sleep(2)

                # edge detection tests
                start = time.time()
                Framework_only_GPU.edge_detection(EDGE_DETECTION_SAMPLE_IMAGE,
                                                  256)
                end = time.time()
                timetable_edgedetection.append((end - start) * 1000)
                time.sleep(2)

            avg_grayscale = str(round(np.average(timetable_grayscale), 3))
            avg_fp32_division = str(
                round(np.average(timetable_operation_division_fp32), 3))
            avg_fp32_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp32), 3))
            avg_fp32_addition = str(
                round(np.average(timetable_operation_addition_fp32), 3))
            avg_fp32_multiply = str(
                round(np.average(timetable_operation_multiply_fp32), 3))
            avg_fp64_division = str(
                round(np.average(timetable_operation_division_fp64), 3))
            avg_fp64_subtraction = str(
                round(np.average(timetable_operation_subtraction_fp64), 3))
            avg_fp64_addition = str(
                round(np.average(timetable_operation_addition_fp64), 3))
            avg_fp64_multiply = str(
                round(np.average(timetable_operation_multiply_fp64), 3))
            avg_edgedetection = str(
                round(np.average(timetable_edgedetection), 3))

            max_grayscale = str(round(max(timetable_grayscale), 3))
            max_fp32_multiply = str(
                round(max(timetable_operation_multiply_fp32), 3))
            max_fp32_addition = str(
                round(max(timetable_operation_addition_fp32), 3))
            max_fp32_subtraction = str(
                round(max(timetable_operation_subtraction_fp32), 3))
            max_fp32_division = str(
                round(max(timetable_operation_division_fp32), 3))
            max_fp64_division = str(
                round(max(timetable_operation_division_fp64), 3))
            max_fp64_multiply = str(
                round(max(timetable_operation_multiply_fp64), 3))
            max_fp64_addition = str(
                round(max(timetable_operation_addition_fp64), 3))
            max_fp64_subtraction = str(
                round(max(timetable_operation_subtraction_fp64), 3))
            max_edgedetection = str(round(max(timetable_edgedetection), 3))

            min_grayscale = str(round(min(timetable_grayscale), 3))
            min_fp32_multiply = str(
                round(min(timetable_operation_multiply_fp32), 3))
            min_fp32_addition = str(
                round(min(timetable_operation_addition_fp32), 3))
            min_fp32_subtraction = str(
                round(min(timetable_operation_subtraction_fp32), 3))
            min_fp32_division = str(
                round(min(timetable_operation_division_fp32), 3))
            min_fp64_division = str(
                round(min(timetable_operation_division_fp64), 3))
            min_fp64_multiply = str(
                round(min(timetable_operation_multiply_fp64), 3))
            min_fp64_addition = str(
                round(min(timetable_operation_addition_fp64), 3))
            min_fp64_subtraction = str(
                round(min(timetable_operation_subtraction_fp64), 3))
            min_edgedetection = str(round(min(timetable_edgedetection), 3))

            res = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Measurements}}
\\label{{tab:performance}}
\\begin{{tabular}}{{|l|l|l|l|}}
\\hline
grayscale & avg: {avg_grayscale} & max: {max_grayscale} & min: {min_grayscale} \\\\
fp32\_division & avg: {avg_fp32_division} & max: {max_fp32_division} & min: {min_fp32_division} \\\\
fp32\_multiply & avg: {avg_fp32_multiply} & max: {max_fp32_multiply} & min: {min_fp32_multiply} \\\\
fp32\_addition & avg: {avg_fp32_addition} & max: {max_fp32_addition} & min: {min_fp32_addition} \\\\
fp32\_subtraction & avg: {avg_fp32_subtraction} & max: {max_fp32_subtraction} & min: {min_fp32_subtraction} \\\\
fp64\_division & avg: {avg_fp64_division} & max: {max_fp64_division} & min: {min_fp64_division} \\\\
fp64\_multiply & avg: {avg_fp64_multiply} & max: {max_fp64_multiply} & min: {min_fp64_multiply} \\\\
fp64\_addition & avg: {avg_fp64_addition} & max: {max_fp64_addition} & min: {min_fp64_addition} \\\\
fp64\_subtraction & avg: {avg_fp64_subtraction} & max: {max_fp64_subtraction} & min: {min_fp64_subtraction} \\\\
edge\_detection & avg: {avg_edgedetection} & max: {max_edgedetection} & min: {min_edgedetection} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            file_results.write(res)
            file_results.close()
print("done..")
