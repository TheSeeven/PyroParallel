from .hardware_resources.OpenCLPlatform import *

VERBOSE = False

RED = "\x1b[31m"
GREEN = "\x1b[32m"
RESET_COLOR = "\x1b[0m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"


def log(*args, end="\n", sep="", colour=RESET_COLOR):
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

    def get_platforms(
        self,
        emulation=False,
        empty_platform=False,
    ):
        self.opencl_platforms = [
            OpenCLPlatform(platform) for platform in OpenCL.get_platforms()
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
                 exclude_others=False):

        global VERBOSE
        VERBOSE = verbose
        self.emulation = emulation
        self.empty_platform = empty_platform
        self.exclude_CPU = exclude_CPU
        self.exclude_GPU = exclude_GPU
        self.exclude_FPGA = exclude_FPGA
        self.exclude_others = exclude_others

        self.get_platforms(emulation=emulation, empty_platform=empty_platform)

        self.opencl_devices = []
        for platform in self.opencl_platforms:
            platform.get_all_devices(exclude_CPU=self.exclude_CPU,
                                     exclude_GPU=self.exclude_GPU,
                                     exclude_FPGA=self.exclude_FPGA,
                                     exclude_others=self.exclude_others)
            for device in platform.devices:
                self.opencl_devices.append(device)
            platform.create_context()

        log("Resources initialised with EMULATION: {0} and EMPTY_PLATFORMS {1}"
            .format(self.emulation, self.empty_platform),
            colour=YELLOW)

        log("Platforms detected: {0}".format(len(self.opencl_platforms)),
            colour=YELLOW)
        counter = 1
        for platform in self.opencl_platforms:
            log("{0}:{1}".format(str(counter), str(platform)), colour=CYAN)
            counter += 1

    def process(self):
        pass

    def benchmark_api(self):
        pass
