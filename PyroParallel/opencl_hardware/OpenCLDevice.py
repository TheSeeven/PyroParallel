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
            CHUNK_PROCESSING_SIZE (int): Controls how much a work-item will do. Larger values yield to lower global_size and local_size. Choose wisely.
        '''
        self.device = device
        self.context = opencl_device.Context([device])
        self.device_name = self.device.name
        self.device_type = self.device.type,
        self.hardware_extensions = self.device.extensions
        self.queue = opencl_device.CommandQueue(
            self.context,
            properties=opencl_device.command_queue_properties.PROFILING_ENABLE)
        self.CHUNK_PROCESSING_SIZE = CHUNK_PROCESSING_SIZE
        self.profiling = {}
        self.max_work_group_size_device = self.device.max_work_group_size

    def _grayscale(self):
        pass

    def _benchmark_grayscale(self):
        # benchmark_img = opencl_device_numpy.random.randint(
        #     0, 256, size=(1000, 10000, 3), dtype=opencl_device_numpy.uint8)
        benchmark_img = opencl_device_numpy.zeros(
            (10000, 1000, 3), dtype=opencl_device_numpy.uint8)
        benchmark_img[:, :, 0] = 255
        opencl_buffers = OpenCLBuffer(
            opencl_device.Buffer(self.context,
                                 opencl_device.mem_flags.READ_ONLY
                                 | opencl_device.mem_flags.USE_HOST_PTR,
                                 hostbuf=benchmark_img),
            opencl_device.Buffer(self.context,
                                 opencl_device.mem_flags.READ_WRITE
                                 | opencl_device.mem_flags.ALLOC_HOST_PTR,
                                 size=benchmark_img.size))
        kernel_program = opencl_device.Program(self.context,
                                               Kernels._GRAYSCALE()).build()
        kernel_program_call = kernel_program.grayscale
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

        output_test = opencl_device_numpy.empty_like(benchmark_img)
        timetable = []
        for _ in range(10):
            event_processing = opencl_device.enqueue_nd_range_kernel(
                self.queue, kernel_program_call, global_size, local_size)
            event_transfer = opencl_device.enqueue_copy(
                self.queue, output_test, opencl_buffers.opencl_buffer_output)

            event_processing.wait()
            event_transfer.wait()
            time = event_transfer.profile.end - event_transfer.profile.start
            timetable.append(time)
        OpenCLFunctions.Pictures.save_array_as_image(output_test,
                                                     "./output_test/")
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