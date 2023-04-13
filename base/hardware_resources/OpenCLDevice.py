class OpenCLDevice:

    def __init__(self, device, device_name, device_type, device_context,
                 hardware_extensions):
        self.device = device
        self.context = device_context
        self.device_name = device_name
        self.device_type = device_type
        self.hardware_extensions = hardware_extensions

    def benchmark_memory(self):
        pass

    def benchmark_instructions(self):
        pass

    def benchmark(self):
        pass

    def __str__(self):
        return self.device_name