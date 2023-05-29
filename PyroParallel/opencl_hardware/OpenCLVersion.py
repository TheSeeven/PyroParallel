# External import
import re as opencl_regex
# External import

# Proprietary import
from .exception.version_exception import *
# Proprietary import


class OpenCLVersion:

    VERSION_EQUAL = 0
    VERSION_GREATER = 1
    VERSION_LOWER = 2

    def _getRegex():
        return r'^OpenCL\s(?P<major>\d+)\.(?P<minor>\d+)(\s(?P<platform>.+))?(?P<remaining>.*)?$'

    def __init__(self, version_str):
        '''__init__ Initialises an OpenCLVersion object based on the imput string. Uses Regex to get major and minor version plus platform.

        Args:
            version_str (string): Parses the string to major version, minor version and platform.

        Raises:
            InvalidOpenCLVersionError: If the string of the version cannot be parsed
            
        '''
        regex = opencl_regex.compile(OpenCLVersion._getRegex())
        match = regex.match(version_str)
        if regex.match(version_str):
            self.major = int(match.group('major'))
            self.minor = int(match.group('minor'))
            self.platform = match.group('platform') or ''
        else:
            raise InvalidOpenCLVersionError(version_str)

    def _compare_to_version(self, version):
        '''compare_to_version Compares two OpenCLVersion objects. The returned result is relative to self, not OpenCLVersion argument.

        Args:
            version (OpenCLVersion): OpenCLVersion object to be compared to.

        Returns:
            int: OpenCLVersion.VERSION_EQUAL = 0
            int: OpenCLVersion.VERSION_GREATER = 1
            int: OpenCLVersion.VERSION_LOWER = 2
        '''
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
