import re as Regex


class InvalidOpenCLVersionError(Exception):

    def __init__(self, version_str):
        self.version_str = version_str
        self.message = f"The OpenCL version string '{version_str}' is invalid."
        super().__init__(self.message)


class OpenCLVersion:

    VERSION_EQUAL = 0
    VERSION_GREATER = 1
    VERSION_LOWER = 2

    def getRegex():
        return r'^OpenCL\s(?P<major>\d+)\.(?P<minor>\d+)(\s(?P<platform>.+))?(?P<remaining>.*)?$'

    def __init__(self, version_str):
        regex = Regex.compile(OpenCLVersion.getRegex())
        match = regex.match(version_str)
        if regex.match(version_str):
            self.major = int(match.group('major'))
            self.minor = int(match.group('minor'))
            self.platform = match.group('platform') or ''
        else:
            raise InvalidOpenCLVersionError(version_str)

    def compare_to_versions(self, version):
        if self.major == version.major:
            if self.minor > version.minor:
                return OpenCLVersion.VERSION_GREATER
            elif self.minor < version.minor:
                return OpenCLVersion.VERSION_LOWER
            return OpenCLVersion.VERSION_EQUAL
        if self.major > version.major:
            return OpenCLVersion.VERSION_GREATER
        elif self.major < version.major:
            return OpenCLVersion.VERSION_LOWER
        return OpenCLVersion.VERSION_EQUAL

    def __str__(self):
        return "OpenCL {0}.{1} {2}".format(str(self.major), str(self.minor),
                                           self.platform)
