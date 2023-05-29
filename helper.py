import numpy as np
from PIL import Image
import pyopencl as cl
import os, psutil
import timeit
import math
import py3nvml.py3nvml as nvml
from main import *


def get_kilobytes(kylobytes):
    return kylobytes * 1024


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


CHUNK_PROCESSING_SIZE = get_kilobytes(16)
pid = os.getpid()
py_process = psutil.Process(pid)
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

kernel_code_grayscale = """
__kernel void grayscale(__global uchar *input,
                        const ulong size)
{
    float luma,r,g,b;
    for(ulong pixel = 0; pixel<(size-2) ; pixel+=3) {
            r = input[pixel+0];
            g = input[pixel+1];
            b = input[pixel+2];

            luma = 0.299f * b + 0.587f * g + 0.114f * r;

            input[pixel+0] = luma;
            input[pixel+1] = luma;
            input[pixel+2] = luma; 
    }
}
"""

#data acquisition and preparation
mem_usage()
Image.MAX_IMAGE_PIXELS = None
picture_path = """C:\\Users\\peria\\Videos\\Mass Effect Andromeda\\25600x14400_1_NO_HDR.bmp"""
im = Image.open(picture_path)
mem_usage()
size = im.width * im.height * 3
mem_usage()
im_arr = np.array_split(
    np.array(im).ravel(), math.ceil(size / CHUNK_PROCESSING_SIZE))
im.close()
mem_usage()
#devices handling
Framework = PyroParallel(verbose=True, exclude_FPGA=True, exclude_others=True)

bufferObjects = []
deviceIndex = 0
for chunk in im_arr:
    buff_obj = cl.Buffer(Framework.opencl_devices[deviceIndex].context,
                         cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                         hostbuf=chunk)
    cl.enqueue_copy(Framework.opencl_devices[deviceIndex].queue, buff_obj,
                    chunk)
    bufferObjects.append(buff_obj)

    deviceIndex += 1
    if deviceIndex >= len(Framework.opencl_devices):
        deviceIndex = 0
mem_usage()

prgs = []
for device in Framework.opencl_devices:
    prgs.append(cl.Program(device.context, kernel_code_grayscale).build())

prgIndex = 0
global_size = (1, 1)
local_size = (1, 1)

for i in range(len(bufferObjects)):
    chunk = bufferObjects[i]
    prgs[prgIndex].grayscale(Framework.opencl_devices[prgIndex].queue,
                             global_size, local_size, chunk,
                             np.uint64(chunk.size))
    prgIndex += 1
    if prgIndex >= len(prgs):
        prgIndex = 0

for device in Framework.opencl_devices:
    device.queue.finish()
print("sssssss")
# im_filtered_arr = np.empty_like(im_arr)
# # Save the filtered image to disk
# im_filtered_arr = Image.fromarray(im_filtered_arr)
# im_filtered_arr.save("grayscale.bmp")