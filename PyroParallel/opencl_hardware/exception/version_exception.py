class InvalidOpenCLVersionError(Exception):

    def __init__(self, version_str):
        self.version_str = version_str
        self.message = f"The OpenCL version string '{version_str}' is invalid."
        super().__init__(self.message)