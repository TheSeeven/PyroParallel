# External import
import pyopencl as opencl_api
# External import

# Proprietary import
from .opencl_hardware.OpenCLPlatform import OpenCLPlatform
from .opencl_hardware.OpenCLDevice import OpenCLDevice
from .opencl_hardware.OpenCLFunctions import OpenCLFunctions
# Proprietary import

VERBOSE = False

RED = "\x1b[31m"
GREEN = "\x1b[32m"
RESET_COLOR = "\x1b[0m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"


def _log(*args, end="\n", sep="", colour=RESET_COLOR):
    if VERBOSE:
        modified_args = []
        for arg in args:
            modified_arg = colour + arg
            modified_arg = modified_arg.replace(
                "False", "{0}False{1}".format(RED, colour))
            modified_arg = modified_arg.replace(
                "True", "{0}True{1}".format(GREEN, colour))

            modified_args.append(modified_arg)
        print(sep.join(modified_args), end=end, sep=sep)
    print(RESET_COLOR, end="", sep="")


class PyroParallel:
    ''' \n
    This API manages hardware resources and allows for the generation and selection of specific groups of devices based on filters. 
    It offers profiling mechanisms and returns platform and device information that can be used outside the PyroParallel context. 
    The API also supports heterogeneous processing of different algorithm implementations.
    '''

    def _get_platforms(self,
                       CHUNKCHUNK_PROCESSING_SIZE,
                       emulation=False,
                       empty_platform=False):
        self.opencl_platforms = [
            OpenCLPlatform(platform, CHUNKCHUNK_PROCESSING_SIZE)
            for platform in opencl_api.get_platforms()
            if (emulation or "EMULATION" not in platform.name.upper()) and (
                empty_platform or len(platform.get_devices()) > 0)
        ]

        return self.opencl_platforms

    def __init__(self,
                 emulation=False,
                 empty_platform=False,
                 verbose=False,
                 exclude_CPU=False,
                 exclude_GPU=False,
                 exclude_FPGA=False,
                 exclude_others=False,
                 CHUNKCHUNK_PROCESSING_SIZE=32):
        '''__init__ \n
            The API object is created by initializing all platforms and devices based on filters. Arguments are optional and by default, all non-virtual devices are generated. 
            NOTE: Profiling is not included in the initialization and must be done afterward. 
            If not, devices are considered to have equal performance. The API learns the performance index of devices in real-time as more processing is performed.
        Args:
            emulation (bool, optional): Defaults to False. If True, it will also generate emulation devices like virtual FPGA.\n
            empty_platform (bool, optional): Defaults to False. If True, Generates platforms that might have no devices available.\n
            verbose (bool, optional):  Defaults to False. If True, text will be displayed about the status of API as it is processing.\n
            exclude_CPU (bool, optional): Defaults to False. If True, no CPUs contexts will be generated.\n
            exclude_GPU (bool, optional): Defaults to False. If True, no GPUs contexts will be generated.\n
            exclude_FPGA (bool, optional): Defaults to False. If True, no FPGAs contexts will be generated.\n
            exclude_others (bool, optional): Defaults to False. If True, no other contexts for devices that don't fit any other criteria will be generated.\n
            exclude_others (int, optional): Defaults to 32. Sets how many bytes shall be processed in one work item.
        '''
        global VERBOSE
        VERBOSE = verbose
        self.emulation = emulation
        self.empty_platform = empty_platform
        self.exclude_CPU = exclude_CPU
        self.exclude_GPU = exclude_GPU
        self.exclude_FPGA = exclude_FPGA
        self.exclude_others = exclude_others
        self.CHUNKCHUNK_PROCESSING_SIZE = CHUNKCHUNK_PROCESSING_SIZE
        self.opencl_platforms = []
        self.opencl_devices = []
        # Create all platform OpenCLPlatform objects
        self._get_platforms(self.CHUNKCHUNK_PROCESSING_SIZE,
                            emulation=emulation,
                            empty_platform=empty_platform)

        # Get all platforms and create context for them
        for platform in self.opencl_platforms:
            platform._get_all_devices(exclude_CPU=self.exclude_CPU,
                                      exclude_GPU=self.exclude_GPU,
                                      exclude_FPGA=self.exclude_FPGA,
                                      exclude_others=self.exclude_others)
            platform._create_context_queue()
            # Get all devices and create contexts for them
            for device in platform.devices:
                self.opencl_devices.append(
                    OpenCLDevice(
                        device=device,
                        CHUNK_PROCESSING_SIZE=self.CHUNKCHUNK_PROCESSING_SIZE))

        _log(
            "Resources initialised with EMULATION: {0} and EMPTY_PLATFORMS {1}"
            .format(self.emulation, self.empty_platform),
            colour=YELLOW)

        _log("Platforms detected: {0}".format(len(self.opencl_platforms)),
             colour=YELLOW)
        counter = 1
        for platform in self.opencl_platforms:
            _log("{0}:{1}".format(str(counter), str(platform)), colour=CYAN)
            counter += 1

    def get_all_devices_contexts(self):
        '''get_all_devices_contexts Returns all contexts for each individual devices. 
            This is usefull if own implementation is needed/wanted

        Returns:
            list(pyopencl.Context): An array having contexts of all individual devices.
            
        '''
        result = []
        if self.opencl_devices:
            if len(self.opencl_devices) > 0:
                result = [device.context for device in self.opencl_devices]
        return result

    def get_all_platform_contexts(self):
        '''get_all_platform_contexts Returns all contexts for each individual platform. 
            This is usefull if own implementation is needed/wanted

        Returns:
            list(pyopencl.Context): An array having contexts of all individual platforms.
        '''
        result = []
        if self.opencl_platforms:
            if len(self.opencl_platforms) > 0:
                result = [
                    platform.context for platform in self.opencl_platforms
                ]
        return result

    def grayscale(self):
        pass

    def double_precision(self):
        pass

    def edge_detection(self):

        pass

    def kilobytes(number):
        '''kilobytes Returns the number of bytes for the given kilobytes number

        Args:
            number (int): Represents the wanted kilobytes

        Returns:
            int: Represents the number of bytes for the given kilobytes
        '''
        return number * (1024 * 1)

    def megabytes(number):
        '''megabytes Returns the number of bytes for the given megabytes number

        Args:
            number (int): Represents the wanted megabytes

        Returns:
            int: Represents the number of bytes for the given megabytes
        '''
        return number * (1024**2)

    def gigabytes(number):
        '''gigabytes Returns the number of bytes for the given gigabytes number

        Args:
            number (int): Represents the wanted gigabytes

        Returns:
            int: Represents the number of bytes for the given gigabytes
        '''
        return number * (1024**3)

    def process(self):
        pass

    def benchmark_api(self):
        '''benchmark_api Creates the performance indexes so that the API when processes something it will know the performance of devices before the processing starts. 
        If this is not executed, equal performance indexes are asssigned to all devices and while processing, the API will learn and adjust the indexes based on the measured performance.
        '''

        for platform in self.opencl_platforms:
            platform._benchmark()
        for device in self.opencl_devices:
            device._benchmark()
        if len(self.opencl_devices) > 0:
            functions = [
                function for function in self.opencl_devices[0].profiling
            ]

            for function in functions:
                times = []
                for device in self.opencl_devices:
                    times.append(device.profiling[function])
                times = OpenCLFunctions.Time.calculate_performance_scores(
                    times)
                for index in range(len(self.opencl_devices)):
                    self.opencl_devices[index].profiling[function] = times[
                        index]
