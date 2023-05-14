import pyopencl as OpenCL
from .OpenCLVersion import OpenCLVersion
from .OpenCLDevice import OpenCLDevice


class OpenCLPlatform:

    def __init__(self, platform):
        self.devices = None
        self.context = None
        self.platform = platform
        self.name = platform.name
        self.version = OpenCLVersion(platform.version)

    def get_all_devices(self,
                        exclude_CPU=False,
                        exclude_GPU=False,
                        exclude_FPGA=False,
                        exclude_others=False):
        filters = []

        if exclude_CPU:
            filters.append(
                lambda device: device.type != OpenCL.device_type.CPU)
        if exclude_GPU:
            filters.append(
                lambda device: device.type != OpenCL.device_type.GPU)
        if exclude_FPGA:
            filters.append(
                lambda device: device.type != OpenCL.device_type.ACCELERATOR)
        if exclude_others:
            filters.append(lambda device: not device.type not in [
                OpenCL.device_type.CPU, OpenCL.device_type.GPU, OpenCL.
                device_type.ACCELERATOR
            ])
        if self.devices is None:
            self.devices = []
            alldevices = self.platform.get_devices()
            for device in alldevices:
                temp = all([filter(device) for filter in filters])
                if temp:
                    context = OpenCL.Context([device])
                    queue = OpenCL.CommandQueue(context)
                    self.devices.append(
                        OpenCLDevice(device=device,
                                     device_context=context,
                                     device_name=device.name,
                                     device_type=device.type,
                                     hardware_extensions=device.extensions,
                                     queue=queue))
        return self.devices

    def create_context(self):
        all_devices = [device.device for device in self.devices]
        if not self.context and len(all_devices) > 0:
            self.context = OpenCL.Context(all_devices)

    def __str__(self):
        res = "Platform {0} with platform version {1} and {2} devices:\n".format(
            self.name, str(self.version), len(self.devices))
        for device in self.devices:
            res = res + "    {0}\n".format(str(device))
        return res
