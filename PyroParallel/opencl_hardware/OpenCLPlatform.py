# External import
import pyopencl as opencl_platform
import numpy as opencl_platform_numpy
# External import

# Proprietary import
from .exception.platform_exception import *
from .OpenCLBuffer import OpenCLBuffer
from .OpenCLFunctions import OpenCLFunctions
from .OpenCLVersion import OpenCLVersion
from .OpenCLKernels import Kernels
# Proprietary import

ENQUEUE_PROCESS_ASYNC = opencl_platform.enqueue_nd_range_kernel
ENQUEUE_COPY_ASYNC = opencl_platform.enqueue_copy
OPENCL_BUFFER = opencl_platform.Buffer
OPENCL_CONTEXT = opencl_platform.Context
OPENCL_QUEUE = opencl_platform.CommandQueue
OPENCL_PROFILING = opencl_platform.command_queue_properties.PROFILING_ENABLE
OPENCL_PROGRAM = opencl_platform.Program
OPENCL_uint8 = opencl_platform_numpy.uint8
OPENCL_uint32 = opencl_platform_numpy.uint32
OPENCL_uint64 = opencl_platform_numpy.uint64
OPENCL_int32 = opencl_platform_numpy.int32
OPENCL_float32 = opencl_platform_numpy.float32
OPENCL_float64 = opencl_platform_numpy.float64
OPENCL_WORK_GROUP_SIZE = opencl_platform.kernel_work_group_info.WORK_GROUP_SIZE
OPENCL_LOCAL_GROUP_SIZE = opencl_platform.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE
READ_WRITE = opencl_platform.mem_flags.READ_WRITE
READ_ONLY = opencl_platform.mem_flags.READ_ONLY
ALLOC_HOST_PTR = opencl_platform.mem_flags.ALLOC_HOST_PTR
NUMPY_EMPTY = opencl_platform_numpy.empty_like
NUMPY_FULL = opencl_platform_numpy.full
NUMPY_PROD = opencl_platform_numpy.prod
NUMPY_AVG = opencl_platform_numpy.average
OPENCL_CPU = opencl_platform.device_type.CPU
OPENCL_GPU = opencl_platform.device_type.GPU
OPENCL_ACCELERATOR = opencl_platform.device_type.ACCELERATOR


class OpenCLPlatform:
    ''' Wrapper around OpenCL providing various pre-implemented functions and profiling capabilities.
    This class serves as an interface to interact with OpenCL platform. It provides functions such as benchmarking,
    grayscale processing, and scheduling. The platform stores profile data and allows for the processing of custom kernels.
    '''

    def __init__(self, platform, CHUNK_PROCESSING_SIZE):
        '''__init__ _summary_

        Args:
            platform (pyopencl.Platform): The object to create the wrap around
            CHUNKCHUNK_PROCESSING_SIZE (int):  Controls how much a work-item will do. Larger values yield to lower global_size and local_size. Choose wisely.
        '''
        self.platform = platform
        self.name = platform.name
        self.hardware_extensions = self.platform.extensions
        self.devices = None
        self.context = None
        self.queue = None
        self.max_work_group_size_platform = None
        self.max_work_items_platform = None
        self.version = OpenCLVersion(platform.version)
        self.CHUNK_PROCESSING_SIZE = CHUNK_PROCESSING_SIZE
        self.profiling = {}
        self.kernels = {}

    def _get_all_devices(self,
                         exclude_CPU=False,
                         exclude_GPU=False,
                         exclude_FPGA=False,
                         exclude_others=False):
        filters = []

        if exclude_CPU:
            filters.append(lambda device: device.type != OPENCL_CPU)
        if exclude_GPU:
            filters.append(lambda device: device.type != OPENCL_GPU)
        if exclude_FPGA:
            filters.append(lambda device: device.type != OPENCL_ACCELERATOR)
        if exclude_others:
            filters.append(lambda device: not device.type not in
                           [OPENCL_CPU, OPENCL_GPU, OPENCL_ACCELERATOR])
        if self.devices is None:
            self.devices = []
            alldevices = self.platform.get_devices()
            for device in alldevices:
                temp = all([filter(device) for filter in filters])
                if temp:
                    self.devices.append(device)

        return self.devices

    def _create_context_queue(self):
        if not self.context and len(self.devices) > 0:
            self.context = OPENCL_CONTEXT(self.devices)
            self.queue = OPENCL_QUEUE(self.context,
                                      properties=OPENCL_PROFILING)

    def _build_kernels(self):
        if self.context:
            self.kernels["_grayscale"] = OPENCL_PROGRAM(
                self.context, Kernels._GRAYSCALE()).build()
            self.kernels["_operation_fp32"] = OPENCL_PROGRAM(
                self.context, Kernels._OPERATION_FP32()).build()
            if "cl_khr_fp64" in self.hardware_extensions:
                self.kernels["_operation_fp64"] = OPENCL_PROGRAM(
                    self.context, Kernels._OPERATION_FP64()).build()

    def _get_max_work_group_size(self):
        for device in self.devices:
            if not self.max_work_group_size_platform:
                self.max_work_group_size_platform = device.max_work_group_size
            if device.max_work_group_size > self.max_work_group_size_platform:
                self.max_work_group_size_platform = device.max_work_group_size

    def _get_max_work_items_size(self):
        for device in self.devices:
            if not self.max_work_items_platform:
                self.max_work_items_platform = device.max_work_item_sizes
            elif NUMPY_PROD(device.max_work_item_sizes) > NUMPY_PROD(
                    self.max_work_items_platform):
                self.max_work_items_platform = device.max_work_item_sizes

    def __str__(self):
        res = "Platform {0} with platform version {1} and {2} devices:\n".format(
            self.name, str(self.version), len(self.devices))
        for device in self.devices:
            res = res + "    {0}\n".format(str(device))
        return res

    def _get_max_work_group(self, program):
        primary_device = None
        for device in self.devices:
            if not primary_device:
                primary_device = device
            elif program.get_work_group_info(
                    OPENCL_WORK_GROUP_SIZE,
                    device) > program.get_work_group_info(
                        OPENCL_WORK_GROUP_SIZE, primary_device):
                primary_device = device

        return program.get_work_group_info(OPENCL_WORK_GROUP_SIZE,
                                           primary_device)

    def _get_prefered_local_size(self, program):
        primary_device = None
        for device in self.devices:
            if not primary_device:
                primary_device = device
            elif program.get_work_group_info(
                    OPENCL_LOCAL_GROUP_SIZE,
                    device) > program.get_work_group_info(
                        OPENCL_LOCAL_GROUP_SIZE, primary_device):
                primary_device = device
        return program.get_work_group_info(OPENCL_LOCAL_GROUP_SIZE,
                                           primary_device)

    def _is_platform_available(self):
        return self.context is not None and len(self.devices) > 0

    def _supports_hardware_extensions(self, instruction):
        return instruction in self.hardware_extensions

### DEVICE FUNCTIONS BEGIN

    def _grayscale(self, input_image):
        result = OpenCLBuffer(
            OPENCL_BUFFER(self.context,
                          READ_WRITE
                          | ALLOC_HOST_PTR,
                          size=input_image.size),
            OPENCL_BUFFER(self.context,
                          READ_WRITE
                          | ALLOC_HOST_PTR,
                          size=input_image.size), NUMPY_EMPTY(input_image))
        result.opencl_fetch_input_event = ENQUEUE_COPY_ASYNC(
            self.queue,
            result.opencl_buffer_input,
            input_image,
            is_blocking=None,
            wait_for=None)

        program = self.kernels["_grayscale"].grayscale
        program.set_args(result.opencl_buffer_input,
                         result.opencl_buffer_output,
                         OPENCL_uint32(input_image.shape[0]),
                         OPENCL_uint64(result.size),
                         OPENCL_uint64(self.CHUNK_PROCESSING_SIZE))
        max_work_group_size_kernel = self._get_max_work_group(program)
        prefered_local_size = self._get_prefered_local_size(program)
        global_size = OpenCLFunctions.OpenCLScheduler._get_global_size(
            input_image.shape[0], input_image.shape[1],
            self.CHUNK_PROCESSING_SIZE)
        local_size, global_size = OpenCLFunctions.OpenCLScheduler._get_optimal_local_global_size(
            global_size, max_work_group_size_kernel,
            self.max_work_group_size_platform, self.max_work_items_platform,
            prefered_local_size)

        result.opencl_input_processing_event = ENQUEUE_PROCESS_ASYNC(
            self.queue, program, global_size, local_size, wait_for=None)
        result.opencl_fetch_result_event = ENQUEUE_COPY_ASYNC(
            self.queue,
            result.result_numpy,
            result.opencl_buffer_output,
            is_blocking=None,
            wait_for=None)

        return result

    def _operation_fp(self, A, B, operation, precision):
        result = OpenCLBuffer([
            OPENCL_BUFFER(self.context,
                          READ_ONLY | ALLOC_HOST_PTR,
                          size=A.size * A.itemsize),
            OPENCL_BUFFER(self.context,
                          READ_ONLY | ALLOC_HOST_PTR,
                          size=B.size * B.itemsize)
        ],
                              OPENCL_BUFFER(self.context,
                                            READ_WRITE | ALLOC_HOST_PTR,
                                            size=A.size * A.itemsize),
                              NUMPY_EMPTY(A))
        result.opencl_fetch_input_event = [
            ENQUEUE_COPY_ASYNC(self.queue,
                               result.opencl_buffer_input[0],
                               A,
                               is_blocking=False,
                               wait_for=None),
            ENQUEUE_COPY_ASYNC(self.queue,
                               result.opencl_buffer_input[1],
                               B,
                               is_blocking=False,
                               wait_for=None)
        ]

        program = self.kernels["_operation_fp" + str(precision)].operation
        program.set_args(result.opencl_buffer_input[0],
                         result.opencl_buffer_input[1],
                         result.opencl_buffer_output,
                         OPENCL_uint64(A.shape[0]),
                         OPENCL_uint64(self.CHUNK_PROCESSING_SIZE),
                         OPENCL_int32(operation))
        max_work_group_size_kernel = self._get_max_work_group(program)
        prefered_local_size = self._get_prefered_local_size(program)
        global_size = OpenCLFunctions.OpenCLScheduler._get_global_size(
            A.size, 1, self.CHUNK_PROCESSING_SIZE)
        local_size, global_size = OpenCLFunctions.OpenCLScheduler._get_optimal_local_global_size(
            global_size, max_work_group_size_kernel,
            self.max_work_group_size_device, self.max_work_items_device,
            prefered_local_size)

        result.opencl_input_processing_event = ENQUEUE_PROCESS_ASYNC(
            self.queue, program, global_size, local_size, wait_for=None)
        result.opencl_fetch_result_event = ENQUEUE_COPY_ASYNC(
            self.queue,
            result.result_numpy,
            result.opencl_buffer_output,
            wait_for=None,
            is_blocking=False)

        return result

### DEVICE FUNCTIONS END

### BENCHMARKS BEGIN

    def _benchmark_grayscale(self):
        benchmark_img = NUMPY_FULL((2500, 2500, 3), 255, dtype=OPENCL_uint8)
        opencl_buffers = OpenCLBuffer(
            OPENCL_BUFFER(self.context,
                          READ_ONLY | ALLOC_HOST_PTR,
                          size=benchmark_img.size),
            OPENCL_BUFFER(self.context,
                          READ_WRITE
                          | ALLOC_HOST_PTR,
                          size=benchmark_img.size), NUMPY_EMPTY(benchmark_img))
        program = self.kernels["_grayscale"].grayscale
        program.set_args(opencl_buffers.opencl_buffer_input,
                         opencl_buffers.opencl_buffer_output,
                         OPENCL_uint32(benchmark_img.shape[0]),
                         OPENCL_uint64(opencl_buffers.size),
                         OPENCL_uint64(self.CHUNK_PROCESSING_SIZE))

        max_work_group_size_kernel = self._get_max_work_group(program)
        prefered_local_size = self._get_prefered_local_size(program)

        global_size = OpenCLFunctions.OpenCLScheduler._get_global_size(
            benchmark_img.shape[0], benchmark_img.shape[1],
            self.CHUNK_PROCESSING_SIZE)
        local_size, global_size = OpenCLFunctions.OpenCLScheduler._get_optimal_local_global_size(
            global_size, max_work_group_size_kernel,
            self.max_work_group_size_platform, self.max_work_items_platform,
            prefered_local_size)

        timetable = []
        for _ in range(10):
            opencl_buffers.opencl_fetch_input_event = ENQUEUE_COPY_ASYNC(
                self.queue, opencl_buffers.opencl_buffer_input, benchmark_img)
            opencl_buffers.opencl_input_processing_event = ENQUEUE_PROCESS_ASYNC(
                self.queue,
                program,
                global_size,
                local_size,
                wait_for=[opencl_buffers.opencl_fetch_input_event])
            opencl_buffers.opencl_fetch_result_event = ENQUEUE_COPY_ASYNC(
                self.queue,
                opencl_buffers.result_numpy,
                opencl_buffers.opencl_buffer_output,
                wait_for=[opencl_buffers.opencl_input_processing_event])
            opencl_buffers.opencl_fetch_input_event.wait()
            opencl_buffers.opencl_input_processing_event.wait()
            opencl_buffers.opencl_fetch_result_event.wait()

            time = opencl_buffers.opencl_fetch_result_event.profile.end - opencl_buffers.opencl_fetch_input_event.profile.start
            timetable.append(time)
        self.profiling['_grayscale'] = NUMPY_AVG(timetable)

    def _benchmark_operation_fp(self):
        TEST_SIZE = 2000
        for operation in self.kernels:
            if "_operation_" in operation:
                FLOAT_BYTES = globals()["OPENCL_float" +
                                        operation[len("_operation_fp"):]]
                program = self.kernels[operation].operation

                max_work_group_size_kernel = self._get_max_work_group(program)
                prefered_local_size = self._get_prefered_local_size(program)
                global_size = OpenCLFunctions.OpenCLScheduler._get_global_size(
                    TEST_SIZE, 1, self.CHUNK_PROCESSING_SIZE)
                local_size, global_size = OpenCLFunctions.OpenCLScheduler._get_optimal_local_global_size(
                    global_size, max_work_group_size_kernel,
                    self.max_work_group_size_platform,
                    self.max_work_items_platform, prefered_local_size)

                fp_benchmark_A = NUMPY_FULL((TEST_SIZE, ),
                                            fill_value=1.7456,
                                            dtype=FLOAT_BYTES)
                fp_benchmark_B = NUMPY_FULL((TEST_SIZE, ),
                                            fill_value=3.1234,
                                            dtype=FLOAT_BYTES)
                opencl_buffer = OpenCLBuffer([
                    OPENCL_BUFFER(
                        self.context,
                        READ_ONLY | ALLOC_HOST_PTR,
                        size=fp_benchmark_A.size * fp_benchmark_A.itemsize),
                    OPENCL_BUFFER(
                        self.context,
                        READ_ONLY | ALLOC_HOST_PTR,
                        size=fp_benchmark_B.size * fp_benchmark_B.itemsize)
                ],
                                             OPENCL_BUFFER(
                                                 self.context,
                                                 READ_WRITE | ALLOC_HOST_PTR,
                                                 size=fp_benchmark_B.size *
                                                 fp_benchmark_B.itemsize),
                                             NUMPY_EMPTY(fp_benchmark_A))

                DIVISION = lambda: program.set_args(
                    opencl_buffer.opencl_buffer_input[0], opencl_buffer.
                    opencl_buffer_input[1], opencl_buffer.opencl_buffer_output,
                    OPENCL_uint64(TEST_SIZE),
                    OPENCL_uint64(self.CHUNK_PROCESSING_SIZE),
                    OPENCL_int32(OpenCLFunctions.DIVISION))

                MULTIPLY = lambda: program.set_args(
                    opencl_buffer.opencl_buffer_input[0], opencl_buffer.
                    opencl_buffer_input[1], opencl_buffer.opencl_buffer_output,
                    OPENCL_uint64(TEST_SIZE),
                    OPENCL_uint64(self.CHUNK_PROCESSING_SIZE),
                    OPENCL_int32(OpenCLFunctions.MULTIPLY))

                SUBTRACT = lambda: program.set_args(
                    opencl_buffer.opencl_buffer_input[0], opencl_buffer.
                    opencl_buffer_input[1], opencl_buffer.opencl_buffer_output,
                    OPENCL_uint64(TEST_SIZE),
                    OPENCL_uint64(self.CHUNK_PROCESSING_SIZE),
                    OPENCL_int32(OpenCLFunctions.SUBTRACT))

                ADDITION = lambda: program.set_args(
                    opencl_buffer.opencl_buffer_input[0], opencl_buffer.
                    opencl_buffer_input[1], opencl_buffer.opencl_buffer_output,
                    OPENCL_uint64(TEST_SIZE),
                    OPENCL_uint64(self.CHUNK_PROCESSING_SIZE),
                    OPENCL_int32(OpenCLFunctions.ADDITION))

                operation_types = {
                    DIVISION: "DIVISION",
                    MULTIPLY: "MULTIPLY",
                    SUBTRACT: "SUBTRACT",
                    ADDITION: "ADDITION"
                }
                for operation_arg_set, operation_name in operation_types.items(
                ):
                    operation_arg_set()
                    timetable = []
                    for _ in range(10):

                        opencl_buffer.opencl_fetch_input_event = [
                            ENQUEUE_COPY_ASYNC(
                                self.queue,
                                opencl_buffer.opencl_buffer_input[0],
                                fp_benchmark_A),
                            ENQUEUE_COPY_ASYNC(
                                self.queue,
                                opencl_buffer.opencl_buffer_input[1],
                                fp_benchmark_B)
                        ]
                        opencl_buffer.opencl_input_processing_event = ENQUEUE_PROCESS_ASYNC(
                            self.queue,
                            program,
                            global_size,
                            local_size,
                            wait_for=opencl_buffer.opencl_fetch_input_event)
                        opencl_buffer.opencl_fetch_result_event = ENQUEUE_COPY_ASYNC(
                            self.queue,
                            opencl_buffer.result_numpy,
                            opencl_buffer.opencl_buffer_output,
                            wait_for=[
                                opencl_buffer.opencl_input_processing_event
                            ])
                        opencl_buffer.opencl_fetch_input_event[0].wait()
                        opencl_buffer.opencl_fetch_input_event[1].wait()
                        opencl_buffer.opencl_input_processing_event.wait()
                        opencl_buffer.opencl_fetch_result_event.wait()

                        time = opencl_buffer.opencl_fetch_result_event.profile.end - opencl_buffer.opencl_fetch_input_event[
                            0].profile.start
                        timetable.append(time)
                    self.profiling[operation + "_" +
                                   operation_name] = NUMPY_AVG(timetable)

    def _benchmark_edge_detection(self):
        pass

    def _benchmark(self):
        if self._is_platform_available():
            self._benchmark_grayscale()
            self._benchmark_operation_fp()
            self._benchmark_edge_detection()


### BENCHMARKS END