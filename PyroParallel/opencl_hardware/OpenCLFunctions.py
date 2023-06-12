# External import
import math as opencl_math
import PIL.Image as opencl_image
import pyopencl as opencl_function
import numpy as numpy_function
# External import

# Proprietary import

# Proprietary import


class OpenCLFunctions:
    '''Collection of OpenCL utility functions and classes.

    This class provides various utility functions and classes for working with OpenCL.
    It includes functions for saving arrays as images, calculating performance scores, and an OpenCL scheduler.

    Attributes:
        Pictures (class): Subclass for working with pictures and saving them as images.
        Time (class): Subclass for time-related calculations.
        OpenCLScheduler (class): Subclass for scheduling OpenCL operations.

    '''

    DIVISION = 0
    MULTIPLY = 1
    ADDITION = 2
    SUBTRACT = 3
    FP32 = 32
    FP64 = 64

    class Operation:
        '''Subclass for performing operations on arrays.
        
        Attributes:
            counter (int): Counter for generating unique txt file names.
        '''

        counter = 0

        def save_array_as_text(array, path, precision):
            '''Save an array as a text file.

            Args:
                array (numpy.ndarray): Array to be saved.
                path (str): Path to save the text file.
            '''
            global counter
            fmt = None
            if precision == OpenCLFunctions.FP32:
                fmt = '%.8f'
            elif precision == OpenCLFunctions.FP64:
                fmt = '%.16f'
            numpy_function.savetxt(
                path + str(OpenCLFunctions.Operation.counter) + ".txt",
                array,
                fmt=fmt)
            OpenCLFunctions.Operation.counter += 1

    class Pictures:
        ''' Subclass for working with pictures and saving them as images.
                
        Attributes:
            counter (int): Counter for generating unique image names.
        '''
        counter = 0

        @staticmethod
        def save_array_as_image(array, path):
            '''save_array_as_image Save an array as an image.

            Args:
                array (numpy.ndarray): Array to be saved as an image.
                path (str): Path to save the image.
            '''
            global counter
            opencl_image.fromarray(array).save(
                path + str(OpenCLFunctions.Pictures.counter) + ".bmp")
            OpenCLFunctions.Pictures.counter += 1

        @staticmethod
        def get_work_amount(task_list):
            result = 0
            COMPLETED = opencl_function.command_execution_status.COMPLETE
            processing_status = None
            copy_status = None
            for task in task_list:
                processing_status = task.opencl_input_processing_event.command_execution_status
                copy_status = task.opencl_fetch_result_event.command_execution_status
                if (processing_status + copy_status) > COMPLETED:
                    result += 1
            return result

    class Time:
        ''' Subclass for time-related calculations.

        This class provides functions for calculating performance scores based on execution times.

        '''

        @staticmethod
        def calculate_performance_scores(execution_times):
            '''calculate_performance_scores Calculate performance scores.

            This function calculates performance scores based on a list of execution times.
            The performance score is calculated as the ratio of the minimum execution time to each execution time.

            Args:
                execution_times (dict): List of execution times.

            Returns:
                dict: Performance scores.
            '''
            min_time = min(execution_times.values())
            performance_scores = {
                device: round(min_time / time, 3)
                for device, time in execution_times.items()
            }
            return performance_scores

    class OpenCLScheduler:
        ''' Subclass for scheduling OpenCL operations.

        This class provides a method for obtaining the optimal local and global sizes for a kernel execution.
        '''

        @staticmethod
        def _get_optimal_local_global_size(minimum_global,
                                           max_work_group_size_kernel,
                                           max_work_group_size_device,
                                           max_work_items,
                                           prefered_local_size):
            '''_get_optimal_local_global_size Get the optimal local and global sizes for a kernel execution.

            This function calculates the optimal local and global sizes for a kernel execution
            based on various constraints and preferences._

            Args:
                minimum_global (tuple): Minimum global size required for the kernel.
                max_work_group_size_kernel (int): Maximum work group size for the kernel.
                max_work_group_size_device (int): Maximum work group size for the device.
                max_work_items (tuple): Maximum work items allowed for the device.
                preferred_local_size (int): Preferred local size by the device.


            Returns:
                tuple: Optimal local size and global size.
            '''
            local_size_0 = 1
            local_size_1 = 1
            global_size_0 = minimum_global[0]
            global_size_1 = minimum_global[1]
            if minimum_global[0] > prefered_local_size or minimum_global[
                    1] > prefered_local_size:
                max_work_items = max_work_items[0] * max_work_items[
                    1] * max_work_items[2]
                rest_global_size_0 = prefered_local_size - (
                    global_size_0 % prefered_local_size)
                rest_global_size_1 = prefered_local_size - (
                    global_size_1 % prefered_local_size)
                global_size_0 += rest_global_size_0
                global_size_1 += rest_global_size_1

                if (global_size_0 * global_size_1) > max_work_items:
                    global_size_0 = minimum_global[0]
                    global_size_1 = minimum_global[1]

                local_size_0 = 1
                local_size_1 = 1
                if (local_size_0 * local_size_1) < max_work_group_size_kernel:
                    local_size_0 = prefered_local_size
                    local_size_1 = prefered_local_size
                    if (local_size_0 *
                            local_size_1) > max_work_group_size_kernel:
                        if global_size_0 > global_size_1:
                            while (local_size_1 *
                                   local_size_0) > max_work_group_size_kernel:
                                local_size_1 = int(local_size_1 / 2)
                        else:
                            while (local_size_0 *
                                   local_size_1) > max_work_group_size_kernel:
                                local_size_0 = int(local_size_0 / 2)
                else:
                    pass
            else:
                rest_global_size_0 = prefered_local_size - (
                    global_size_0 % prefered_local_size)
                rest_global_size_1 = prefered_local_size - (
                    global_size_1 % prefered_local_size)
                global_size_0 += rest_global_size_0
                global_size_1 += rest_global_size_1
                local_size_0 = prefered_local_size
                local_size_1 = prefered_local_size
            if (local_size_0 * local_size_1) > max_work_group_size_device:

                if local_size_0 > local_size_1:
                    while (local_size_0 *
                           local_size_1) > max_work_group_size_device:
                        local_size_0 = int(local_size_0 / 2)
                else:
                    while (local_size_1 *
                           local_size_0) > max_work_group_size_device:
                        local_size_1 = int(local_size_1 / 2)
            return (local_size_0, local_size_1), (global_size_0, global_size_1)

        @staticmethod
        def _get_global_size(width, height, chunk_size):
            '''_get_global_size_picture Calculate the global size for an image processing operation.

            This function calculates the global size for an image processing operation based on the width, height,
            and chunk size.

            Args:
                width (int): Width of the image.
                height (int): Height of the image.
                chunk_size (int): Size of the chunks for processing.

            Returns:
                tuple: Global size for the image processing operation.
            '''
            return (opencl_math.ceil(width / chunk_size), height)