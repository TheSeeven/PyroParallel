import numpy as np
from PIL import Image
import pyopencl as cl
import os, psutil
import timeit
import math
import py3nvml.py3nvml as nvml
from main import *


class Buffer_Grayscale:

    counter = 0

    def __init__(self,
                 opencl_buffer,
                 width,
                 height,
                 opencl_buffer_output=None,
                 picture=None):
        self.picture_input = picture
        self.picture_output = np.empty_like(self.picture_input)
        self.width = width
        self.height = height
        self.size = opencl_buffer.size
        self.opencl_buffer_input = opencl_buffer
        self.opencl_buffer_output = opencl_buffer_output

    def save_output_on_disk(self, path, queue=None):
        if type(self.opencl_buffer_output) is not cl.Buffer:
            Image.fromarray(
                self.opencl_buffer_output).save(path + "\\" +
                                                str(Buffer_Grayscale.counter) +
                                                ".bmp")
        else:
            cl.enqueue_copy(self.queue, self.picture_output,
                            self.opencl_buffer_output)
            Image.fromarray(
                self.picture_output).save(path + "\\" +
                                          str(Buffer_Grayscale.counter) +
                                          ".bmp")
        Buffer_Grayscale.counter += 1


### Config
CHUNK_PROCESSING_SIZE = 128
INPUT_PATH = "C:\\Users\\peria\\Desktop\\images gathering\\input"
OUTPUT_PATH = "C:\\Users\\peria\\Desktop\\images gathering\\output"
pid = os.getpid()
py_process = psutil.Process(pid)
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)
Image.MAX_IMAGE_PIXELS = None
### Config


def mem_usage():
    info = nvml.nvmlDeviceGetMemoryInfo(handle)
    vram_usage_mb = info.used / (1024**2)
    print("*******")
    print("RAM usage: ", py_process.memory_info().rss / (1024**2), "MB")
    print("VRAM usage: ", vram_usage_mb, "MB")
    print("*******")


def get_global_size(width, height, chunk_size):
    result = None
    if width > 0 and height > 0:
        total_size = width * height
        number_of_chunks = math.ceil(total_size / chunk_size)
        if width > height:
            chunks_per_row = int(width / chunk_size)
            if chunks_per_row <= 0:
                chunks_per_row = 1
            chunks_per_column = math.ceil(number_of_chunks / chunks_per_row)
            result = (chunks_per_row, chunks_per_column)
        else:
            chunks_per_column = int(height / chunk_size)
            if chunks_per_column <= 0:
                chunks_per_column = 1
            chunks_per_row = math.ceil(number_of_chunks / chunks_per_column)
            result = (chunks_per_row, chunks_per_column)
    return result


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
kernel_code_grayscale = """
__kernel void grayscale(__global uchar *input,
                        __global uchar *output,
                        uint width,
                        ulong size,
                        const ulong chunk_size)
{
    ulong gid_column = get_global_id(0);
    ulong gid_row = get_global_id(1);
    ulong pixel;
    float luma,r,g,b;
    for(ulong i = 0; i<chunk_size;i++){
        pixel = ((gid_column*chunk_size)+(gid_row*width)+i)*3;
        if(pixel<size){
            r = input[pixel+0];
            g = input[pixel+1];
            b = input[pixel+2];
            luma = 0.299f * b + 0.587f * g + 0.114f * r;
            output[pixel+0] = luma;
            output[pixel+1] = luma;
            output[pixel+2] = luma;
        }
        else break;
    }
    
}
"""

open_pictures = [
    Image.open(INPUT_PATH + "\\" + picture)
    for picture in os.listdir(INPUT_PATH)
]

input_pictures_array = [np.array(picture) for picture in open_pictures]
output_pictures_array = [
    np.empty_like(picture) for picture in input_pictures_array
]

opencl_ctx = cl.create_some_context()
opencl_queue = cl.CommandQueue(opencl_ctx)

# Create a buffer object in OpenCL with the numpy array data
opencl_input_buffers = [
    Buffer_Grayscale(
        cl.Buffer(opencl_ctx,
                  cl.mem_flags.READ_ONLY
                  | cl.mem_flags.COPY_HOST_PTR,
                  hostbuf=picture), picture.shape[1], picture.shape[0],
        cl.Buffer(opencl_ctx,
                  cl.mem_flags.WRITE_ONLY
                  | cl.mem_flags.ALLOC_HOST_PTR,
                  size=picture.size), picture)
    for picture in input_pictures_array
]

timer_list = []

# Create the OpenCL program and kernel
prg = cl.Program(opencl_ctx, kernel_code_grayscale).build()
grayscale_kernel = prg.grayscale
local_size = (1, 1)
for i in range(10):
    start_time = timeit.default_timer()
    for opencl_buffer in opencl_input_buffers:
        global_size = (math.ceil(opencl_buffer.width / CHUNK_PROCESSING_SIZE),
                       math.ceil(opencl_buffer.height))
        grayscale_kernel.set_args(opencl_buffer.opencl_buffer_input,
                                  opencl_buffer.opencl_buffer_output,
                                  np.uint32(opencl_buffer.width),
                                  np.uint64(opencl_buffer.size),
                                  np.uint64(CHUNK_PROCESSING_SIZE))
        cl.enqueue_nd_range_kernel(opencl_queue, grayscale_kernel, global_size,
                                   local_size)
        cl.enqueue_copy(opencl_queue, opencl_buffer.picture_output,
                        opencl_buffer.opencl_buffer_output)
    opencl_queue.finish()
    end_time = timeit.default_timer()
    timer_list.append(end_time - start_time)

print("All values:\n")
print(" | ", end="")
for i in timer_list:
    print("{:.3f}".format(i), end=" | ")
print("\n")
print("max: " + str(max(timer_list)))
print("min: " + str(min(timer_list)))
print("mean: " + str(np.average(timer_list)))
for filtered_picture in opencl_input_buffers:
    filtered_picture.save_output_on_disk(OUTPUT_PATH, queue=opencl_queue)

# Save the filtered image to disk

print("done")