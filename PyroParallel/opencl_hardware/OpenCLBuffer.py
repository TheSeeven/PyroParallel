class OpenCLBuffer:

    def __init__(self, opencl_buffer_input, opencl_buffer_output):
        self.opencl_buffer_input = opencl_buffer_input
        self.opencl_buffer_output = opencl_buffer_output
        self.size = opencl_buffer_output.size
