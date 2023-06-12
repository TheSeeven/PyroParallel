# External import
import pyopencl as opencl_device
import numpy as opencl_device_numpy
# External import

# Proprietary import
from .exception.device_exception import *
from .OpenCLBuffer import OpenCLBuffer
from .OpenCLFunctions import OpenCLFunctions
from .OpenCLKernels import Kernels
# Proprietary import

ENQUEUE_PROCESS_ASYNC = opencl_device.enqueue_nd_range_kernel
ENQUEUE_COPY_ASYNC = opencl_device.enqueue_copy
OPENCL_BUFFER = opencl_device.Buffer
OPENCL_CONTEXT = opencl_device.Context
OPENCL_QUEUE = opencl_device.CommandQueue
OPENCL_PROFILING = opencl_device.command_queue_properties.PROFILING_ENABLE
OPENCL_PROGRAM = opencl_device.Program
OPENCL_uint8 = opencl_device_numpy.uint8
OPENCL_uint32 = opencl_device_numpy.uint32
OPENCL_uint64 = opencl_device_numpy.uint64
OPENCL_int32 = opencl_device_numpy.int32
OPENCL_float32 = opencl_device_numpy.float32
OPENCL_float64 = opencl_device_numpy.float64
OPENCL_WORK_GROUP_SIZE = opencl_device.kernel_work_group_info.WORK_GROUP_SIZE
OPENCL_LOCAL_GROUP_SIZE = opencl_device.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE
READ_WRITE = opencl_device.mem_flags.READ_WRITE
READ_ONLY = opencl_device.mem_flags.READ_ONLY
ALLOC_HOST_PTR = opencl_device.mem_flags.ALLOC_HOST_PTR
NUMPY_EMPTY = opencl_device_numpy.empty_like
NUMPY_FULL = opencl_device_numpy.full


class OpenCLDevice:
    ''' Wrapper around an OpenCL device providing various functions and profiling capabilities.This class serves as an interface to interact with an OpenCL device. It provides functions such as benchmarking,
    memory management, kernel execution, and profiling capabilities.
    '''

    def __init__(self, device, CHUNK_PROCESSING_SIZE):
        '''__init__ initialises a device, creates a context, and the created object is then used in conjunction with other objects for performance profiling and encapsulating the idea of a device can execute a task via a simple function call.

        Args:
            device (pyopencl.Device): Creates a wraper around pyopencl.Device.
            CHUNK_PROCESSING_SIZE (int): Controls how much a work-item will do. Larger values yield to lower global_size and local_size. If a device does not support to many work items, increase this value.
        '''
        self.device = device
        self.context = OPENCL_CONTEXT([device])
        self.device_name = self.device.name
        self.device_type = self.device.type,
        self.hardware_extensions = self.device.extensions
        self.queue = OPENCL_QUEUE(
            self.context,
            properties=OPENCL_PROFILING,
        )
        self.CHUNK_PROCESSING_SIZE = CHUNK_PROCESSING_SIZE
        self.max_work_group_size_device = self.device.max_work_group_size
        self.profiling = {}
        self.kernels = {}
        self.kernels["_grayscale"] = OPENCL_PROGRAM(
            self.context, Kernels._GRAYSCALE()).build()
        self.kernels["_operation_fp32"] = OPENCL_PROGRAM(
            self.context, Kernels._OPERATION_FP32()).build()
        if "cl_khr_fp64" in self.hardware_extensions:
            self.kernels["_operation_fp64"] = OPENCL_PROGRAM(
                self.context, Kernels._OPERATION_FP64()).build()

    def _get_max_work_group(self, program):
        return program.get_work_group_info(OPENCL_WORK_GROUP_SIZE, self.device)

    def _get_prefered_local_size(self, program):
        return program.get_work_group_info(OPENCL_LOCAL_GROUP_SIZE,
                                           self.device)

    def _supports_hardware_extensions(self, instruction):
        return instruction in self.hardware_extensions

    def __str__(self):
        return self.device_name

### BENCHMARK BEGIN

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
            self.device.max_work_group_size, self.device.max_work_item_sizes,
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
        self.profiling['_grayscale'] = opencl_device_numpy.average(timetable)

    def _benchmark_operation_fp(self):
        TEST_SIZE = 1000
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
                    self.device.max_work_group_size,
                    self.device.max_work_item_sizes, prefered_local_size)

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
                    self.profiling[
                        operation + "_" +
                        operation_name] = opencl_device_numpy.average(
                            timetable)

    def _benchmark_edge_detection(self):
        pass

    def _benchmark(self):
        self._benchmark_grayscale()
        self._benchmark_operation_fp()
        self._benchmark_edge_detection()
### BENCHMARK END

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
            self.device.max_work_group_size, self.device.max_work_item_sizes,
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
        max_work_group_size_kernel = program.get_work_group_info(
            opencl_device.kernel_work_group_info.WORK_GROUP_SIZE, self.device)
        prefered_local_size = program.get_work_group_info(
            opencl_device.kernel_work_group_info.
            PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.device)
        global_size = OpenCLFunctions.OpenCLScheduler._get_global_size(
            A.size, 1, self.CHUNK_PROCESSING_SIZE)
        local_size, global_size = OpenCLFunctions.OpenCLScheduler._get_optimal_local_global_size(
            global_size, max_work_group_size_kernel,
            self.device.max_work_group_size, self.device.max_work_item_sizes,
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


###DEVICE FUNCTIONS END
