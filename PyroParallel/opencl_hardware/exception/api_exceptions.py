class UnsuportedFunctionality(Exception):

    def __init__(self, func, extensions):
        separator = " | "
        self.message = "The current devices from API does not have the required instructions for the function: " + str(
            func) + ".\n The required extensions are: " + separator.join(
                extensions)
        super().__init__(self.message)


class OperationUnsuported(Exception):

    def __init__(self, operation):
        self.message = "The operation " + str(operation) + " is not supported"
        super().__init__(self.message)


class ResourceAvailability(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ParameterError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ApiStateError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)