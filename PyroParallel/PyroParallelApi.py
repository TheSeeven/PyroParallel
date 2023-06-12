# External import
import pyopencl as opencl_api
# External import

# Proprietary import
from .opencl_hardware.OpenCLPlatform import OpenCLPlatform
from .opencl_hardware.OpenCLDevice import OpenCLDevice
from .opencl_hardware.OpenCLFunctions import OpenCLFunctions
from .opencl_hardware.exception.api_exceptions import *
# Proprietary import

VERBOSE = False

RED = "\x1b[31m"
GREEN = "\x1b[32m"
RESET_COLOR = "\x1b[0m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"

OPENCL_GET_PLATFORMS = opencl_api.get_platforms


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

    DIVISION = 0
    MULTIPLY = 1
    ADDITION = 2
    SUBTRACT = 3
    FP32 = 32
    FP64 = 64

    def _get_platforms(self,
                       CHUNKCHUNK_PROCESSING_SIZE,
                       emulation=False,
                       empty_platform=False):
        self.opencl_platforms = [
            OpenCLPlatform(platform, CHUNKCHUNK_PROCESSING_SIZE)
            for platform in OPENCL_GET_PLATFORMS()
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
                 CHUNKCHUNK_PROCESSING_SIZE=1):
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
            exclude_others (int, optional): Defaults to 1. Sets how many bytes shall be processed in one work item. Increase this value if the maximum work item of a device is to low compared to the problem size
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

    def _get_operation_name(prefix, operation):
        result = None

        if operation == PyroParallel.ADDITION:
            result = prefix + "ADDITION"
        elif operation == PyroParallel.SUBTRACT:
            result = prefix + "SUBTRACT"
        elif operation == PyroParallel.MULTIPLY:
            result = prefix + "MULTIPLY"
        elif operation == PyroParallel.DIVISION:
            result = prefix + "DIVISION"

        return result

    def _get_operation_precision_name(prefix, precision):
        result = None
        if precision == PyroParallel.FP32:
            result = prefix + "fp32_"
        elif precision == PyroParallel.FP64:
            result = prefix + "fp64_"

        return result

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

    @staticmethod
    def kilobytes(number):
        '''kilobytes Returns the number of bytes for the given kilobytes number

        Args:
            number (int): Represents the wanted kilobytes

        Returns:
            int: Represents the number of bytes for the given kilobytes
        '''
        return number * (1024 * 1)

    def grayscale(self, images, autosave=False):
        '''grayscale Applies grayscale on the provided images. It is highly recomanded for performance reasons that the size of images to be identical.

        Args:
            images (numpy.ndarray): This parameter have more than 0 pictures an must be of type numpy matrix.

        Returns:
            numpy.ndarray: Returned results have the same length as the input, the same datatype but with grayscale applied
        '''
        result = None
        if len(images) > 0:
            required_extensions = []
            supported_devices = [
                device for device in self.opencl_devices if all([
                    device._supports_hardware_extensions(extension)
                    for extension in required_extensions
                ])
            ]
            if len(supported_devices) > 0:
                result = []
                device_queues = {}
                device_queues_status = {}

                selected_device = None
                for device in supported_devices:
                    device_queues[device] = []
                    device_queues_status[device] = 0
                device_queues = dict(
                    sorted(device_queues.items(),
                           key=lambda x: -x[0].profiling["_grayscale"]))
                for image in images:
                    for device in device_queues:
                        device_queues_status[
                            device] = OpenCLFunctions.Pictures.get_work_amount(
                                device_queues[device])
                    selected_device = min(
                        device_queues_status,
                        key=device_queues_status.get)  # type: ignore
                    device_queues[selected_device].append(
                        selected_device._grayscale(image))
                for device, events in device_queues.items():
                    for event in events:
                        event.opencl_fetch_result_event.wait()
                        result.append(event.result_numpy)
                if autosave:
                    for output in result:
                        OpenCLFunctions.Pictures.save_array_as_image(
                            output, "./output_test/")
            else:
                raise UnsuportedFunctionality("grayscale", required_extensions)
        return result

    def operation(self, A, B, operation, precision, autosave=False):
        '''operation_fp32 Applies an operation on A and B and returns the result. PyroParallel will try to use all the available devices to perform this operation (if suported). 
        If no devices can perform thiss operation, none is returned. If FP64 precision is used, only devices with cl_khr_fp64 will be used.

        Args:
            A (numpy.ndarray): The first parameter for the operation
            B (numpy.ndarray): The second parameter for the operation
            operation (integer): The operation to be performed on A and B
                                DIVIDE = 0
                                MULTIPLY = 1
                                ADD = 2
                                SUBTRACT = 3
            precision (integer): Specifies the precision in bytes.

        Returns:
            numpy.ndarray: This is the final result after aplying the desired operation the the two inputs
        '''
        result = None
        if not isinstance(A, list) or not isinstance(B, list):
            raise ParameterError(
                "Type of arg1 and arg2 not correct, got: arg1: {0} arg2: {1} ".
                format(str(type(A)), str(type(B))))
        if not ((precision == PyroParallel.FP64) or
                (precision == PyroParallel.FP32)):
            raise ParameterError(
                "The precision value is invalid, got: precision: {0}".format(
                    str(precision)))
        if len(A) == len(B):
            required_extensions = []
            if precision == PyroParallel.FP64:
                required_extensions.append('cl_khr_fp64')
            supported_devices = [
                device for device in self.opencl_devices if all([
                    device._supports_hardware_extensions(extension)
                    for extension in required_extensions
                ])
            ]
            if len(supported_devices) > 0:
                operation_name = PyroParallel._get_operation_name(
                    PyroParallel._get_operation_precision_name(
                        "_operation_", precision), operation)
                if operation_name is not None:
                    result = []
                    input_length = len(A)
                    device_queues = {}
                    device_queues_status = {}

                    temp_A = None
                    temp_B = None

                    selected_device = None
                    for device in supported_devices:
                        device_queues[device] = []
                        device_queues_status[device] = 0
                    device_queues = dict(
                        sorted(device_queues.items(),
                               key=lambda x: -x[0].profiling[operation_name]))
                    for input_index in range(input_length):
                        temp_A = A[input_index]
                        temp_B = B[input_index]
                        for device in device_queues:
                            device_queues_status[
                                device] = OpenCLFunctions.Pictures.get_work_amount(  # create a function specific for operations since it has more ev queues
                                    device_queues[device])
                        selected_device = min(
                            device_queues_status,
                            key=device_queues_status.get)  # type: ignore
                        device_queues[selected_device].append(
                            selected_device._operation_fp(
                                temp_A, temp_B, operation, precision))
                    for device, events in device_queues.items():
                        for event in events:
                            event.opencl_fetch_result_event.wait()
                            result.append(event.result_numpy)
                    if autosave:
                        for output in result:
                            OpenCLFunctions.Operation.save_array_as_text(
                                output, "./output_test/", precision)
                else:
                    raise OperationUnsuported(operation)
            else:
                raise UnsuportedFunctionality("operation", required_extensions)
        else:
            raise ParameterError(
                "Parameter arg1 and parameter arg2 do not have the same sizes, got: arg1 size: {0} arg2 size {1}"
                .format(str(len(A)), str(len(B))))
        return result

    def edge_detection(self):
        pass

    def benchmark_api(self):
        '''benchmark_api Creates the performance indexes so that the API when processes functions it will know the performance of devices before the processing starts. 
        If this is not executed, equal performance indexes are asssigned to all devices and while processing, the API will learn and adjust the indexes based on the measured performance.
        '''

        for platform in self.opencl_platforms:
            platform._benchmark()
        for device in self.opencl_devices:
            device._benchmark()
        if len(self.opencl_devices) > 0:
            functions = set(function_name for device in self.opencl_devices
                            for function_name in device.profiling)

            for function in functions:
                times = {}
                for device_index in range(len(self.opencl_devices)):
                    device = self.opencl_devices[device_index]
                    try:
                        times[device] = device.profiling[function]
                    except:
                        pass
                times = OpenCLFunctions.Time.calculate_performance_scores(
                    times)
                for device in self.opencl_devices:
                    try:
                        device.profiling[function] = times[device]
                    except:
                        pass
        else:
            raise ResourceAvailability(
                "No hardware resources available for framework usage, 0 devices initialised"
            )
