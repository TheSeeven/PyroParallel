class OpenCLBuffer:

    def __init__(self, opencl_buffer_input, opencl_buffer_output,
                 result_numpy):
        self.opencl_buffer_input = opencl_buffer_input
        self.opencl_buffer_output = opencl_buffer_output
        self.opencl_fetch_input_event = None
        self.opencl_input_processing_event = None
        self.opencl_fetch_result_event = None
        self.size = opencl_buffer_output.size
        self.result_numpy = result_numpy
