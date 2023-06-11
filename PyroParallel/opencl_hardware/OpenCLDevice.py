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
        self.context = opencl_device.Context([device])
        self.device_name = self.device.name
        self.device_type = self.device.type,
        self.hardware_extensions = self.device.extensions
        self.queue = opencl_device.CommandQueue(
            self.context,
            properties=opencl_device.command_queue_properties.PROFILING_ENABLE,
        )
        self.CHUNK_PROCESSING_SIZE = CHUNK_PROCESSING_SIZE
        self.max_work_group_size_device = self.device.max_work_group_size
        self.profiling = {}
        self.kernels = {}

    def _grayscale(self, input_image):
        ENQUEUE_PROCESS_ASYNC = opencl_device.enqueue_nd_range_kernel
        ENQUEUE_COPY_ASYNC = opencl_device.enqueue_copy
        OPENCL_BUFFER = opencl_device.Buffer

        result = OpenCLBuffer(
            OPENCL_BUFFER(self.context,
                          opencl_device.mem_flags.READ_WRITE
                          | opencl_device.mem_flags.ALLOC_HOST_PTR,
                          size=input_image.size),
            OPENCL_BUFFER(self.context,
                          opencl_device.mem_flags.READ_WRITE
                          | opencl_device.mem_flags.ALLOC_HOST_PTR,
                          size=input_image.size),
            opencl_device_numpy.empty_like(input_image))
        result.opencl_fetch_input_event = ENQUEUE_COPY_ASYNC(
            self.queue, result.opencl_buffer_input, input_image)

        program = self.kernels["_grayscale"].grayscale
        program.set_args(
            result.opencl_buffer_input, result.opencl_buffer_output,
            opencl_device_numpy.uint32(input_image.shape[0]),
            opencl_device_numpy.uint64(result.size),
            opencl_device_numpy.uint64(self.CHUNK_PROCESSING_SIZE))
        max_work_group_size_kernel = program.get_work_group_info(
            opencl_device.kernel_work_group_info.WORK_GROUP_SIZE, self.device)
        prefered_local_size = program.get_work_group_info(
            opencl_device.kernel_work_group_info.
            PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.device)
        global_size = OpenCLFunctions.Pictures._get_global_size_picture(
            input_image.shape[0], input_image.shape[1],
            self.CHUNK_PROCESSING_SIZE)
        local_size, global_size = OpenCLFunctions.OpenCLScheduler._get_optimal_local_global_size(
            global_size, max_work_group_size_kernel,
            self.device.max_work_group_size, self.device.max_work_item_sizes,
            prefered_local_size)

        result.opencl_input_processing_event = ENQUEUE_PROCESS_ASYNC(
            self.queue,
            program,
            global_size,
            local_size,
            wait_for=[result.opencl_fetch_input_event])
        result.opencl_fetch_result_event = ENQUEUE_COPY_ASYNC(
            self.queue,
            result.result_numpy,
            result.opencl_buffer_output,
            is_blocking=None)

        return result

    def _benchmark_grayscale(self):
        ENQUEUE_PROCESSING = opencl_device.enqueue_nd_range_kernel
        ENQUEUE_COPY = opencl_device.enqueue_copy
        benchmark_img = opencl_device_numpy.full(
            (2500, 2500, 3), 255, dtype=opencl_device_numpy.uint8)
        opencl_buffers = OpenCLBuffer(
            opencl_device.Buffer(self.context,
                                 opencl_device.mem_flags.READ_ONLY
                                 | opencl_device.mem_flags.ALLOC_HOST_PTR,
                                 size=benchmark_img.size),
            opencl_device.Buffer(self.context,
                                 opencl_device.mem_flags.READ_WRITE
                                 | opencl_device.mem_flags.ALLOC_HOST_PTR,
                                 size=benchmark_img.size),
            opencl_device_numpy.empty_like(benchmark_img))
        self.kernels["_grayscale"] = opencl_device.Program(
            self.context, Kernels._GRAYSCALE()).build()
        kernel_program_call = self.kernels["_grayscale"].grayscale
        kernel_program_call.set_args(
            opencl_buffers.opencl_buffer_input,
            opencl_buffers.opencl_buffer_output,
            opencl_device_numpy.uint32(benchmark_img.shape[0]),
            opencl_device_numpy.uint64(opencl_buffers.size),
            opencl_device_numpy.uint64(self.CHUNK_PROCESSING_SIZE))

        max_work_group_size_kernel = kernel_program_call.get_work_group_info(
            opencl_device.kernel_work_group_info.WORK_GROUP_SIZE, self.device)
        prefered_local_size = kernel_program_call.get_work_group_info(
            opencl_device.kernel_work_group_info.
            PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.device)

        global_size = OpenCLFunctions.Pictures._get_global_size_picture(
            benchmark_img.shape[0], benchmark_img.shape[1],
            self.CHUNK_PROCESSING_SIZE)
        local_size, global_size = OpenCLFunctions.OpenCLScheduler._get_optimal_local_global_size(
            global_size, max_work_group_size_kernel,
            self.device.max_work_group_size, self.device.max_work_item_sizes,
            prefered_local_size)

        timetable = []
        for _ in range(10):
            opencl_buffers.opencl_fetch_input_event = ENQUEUE_COPY(
                self.queue, opencl_buffers.opencl_buffer_input, benchmark_img)
            opencl_buffers.opencl_input_processing_event = ENQUEUE_PROCESSING(
                self.queue,
                kernel_program_call,
                global_size,
                local_size,
                wait_for=[opencl_buffers.opencl_fetch_input_event])
            opencl_buffers.opencl_fetch_result_event = ENQUEUE_COPY(
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

    def _benchmark_double_precision(self):
        pass

    def _benchmark_edge_detection(self):
        pass

    def _benchmark(self):
        self._benchmark_grayscale()
        self._benchmark_double_precision()
        self._benchmark_edge_detection()

    def __str__(self):
        return self.device_name