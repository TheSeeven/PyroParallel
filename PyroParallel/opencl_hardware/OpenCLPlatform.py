# External import
import pyopencl as opencl_platform
import numpy as opencl_platform_numpy
# External import

# Proprietary import
from .exception.platform_exception import *
from .OpenCLKernels import Kernels
from .OpenCLVersion import OpenCLVersion
from .OpenCLFunctions import OpenCLFunctions
# Proprietary import


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
        self.devices = None
        self.context = None
        self.queue = None
        self.platform = platform
        self.name = platform.name
        self.version = OpenCLVersion(platform.version)
        self.CHUNK_PROCESSING_SIZE = CHUNK_PROCESSING_SIZE

    def _get_all_devices(self,
                         exclude_CPU=False,
                         exclude_GPU=False,
                         exclude_FPGA=False,
                         exclude_others=False):
        filters = []

        if exclude_CPU:
            filters.append(
                lambda device: device.type != opencl_platform.device_type.CPU)
        if exclude_GPU:
            filters.append(
                lambda device: device.type != opencl_platform.device_type.GPU)
        if exclude_FPGA:
            filters.append(lambda device: device.type != opencl_platform.
                           device_type.ACCELERATOR)
        if exclude_others:
            filters.append(lambda device: not device.type not in [
                opencl_platform.device_type.CPU, opencl_platform.device_type.
                GPU, opencl_platform.device_type.ACCELERATOR
            ])
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
            self.context = opencl_platform.Context(self.devices)
            self.queue = opencl_platform.CommandQueue(self.context)

    def __str__(self):
        res = "Platform {0} with platform version {1} and {2} devices:\n".format(
            self.name, str(self.version), len(self.devices))
        for device in self.devices:
            res = res + "    {0}\n".format(str(device))
        return res

    def _benchmark_grayscale(self):
        pass

    def _benchmark_double_precision(self):
        pass

    def _benchmark_edge_detection(self):
        pass

    def _benchmark(self):
        pass