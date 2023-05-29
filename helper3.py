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
                 opencl_buffer_input,
                 width,
                 height,
                 opencl_buffer_output=None,
                 picture=None,
                 queue=None):
        self.opencl_buffer_input = opencl_buffer_input
        self.opencl_buffer_output = opencl_buffer_output
        self.picture_input = picture
        self.picture_output = np.empty_like(self.picture_input)
        self.size = opencl_buffer_output.size
        self.width = width
        self.height = height
        self.queue = queue

    def save_output_on_disk(self, path):
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


class DeviceSelector:

    def __init__(self, queue, context, device_index):
        self.queue = queue
        self.context = context
        self.device_index = device_index


### Config
CHUNK_PROCESSING_SIZE = 32
INPUT_PATH = "C:\\Users\\peria\\Desktop\\images gathering\\input"
OUTPUT_PATH = "C:\\Users\\peria\\Desktop\\images gathering\\output"
pid = os.getpid()
py_process = psutil.Process(pid)
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)
Image.MAX_IMAGE_PIXELS = None
### Config

CONTEXT_COUNT = len(Framework.opencl_devices)
CURENT_DEVICE = 0
CURRENT_SELECTOR = None


def get_context_queue():
    global CURENT_DEVICE, CONTEXT_COUNT, CURRENT_SELECTOR
    CURRENT_SELECTOR = DeviceSelector(
        Framework.opencl_devices[CURENT_DEVICE].queue,
        Framework.opencl_devices[CURENT_DEVICE].context, CURENT_DEVICE)
    CURENT_DEVICE += 1
    if CURENT_DEVICE >= CONTEXT_COUNT:
        CURENT_DEVICE = 0
    return CURRENT_SELECTOR


def reset_selector():
    global CURENT_DEVICE
    CURENT_DEVICE = 0


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

# Create a buffer object in OpenCL with the numpy array data
opencl_input_buffers = [
    Buffer_Grayscale(
        cl.Buffer(get_context_queue().context,
                  cl.mem_flags.READ_ONLY
                  | cl.mem_flags.USE_HOST_PTR,
                  hostbuf=picture), picture.shape[1], picture.shape[0],
        cl.Buffer(CURRENT_SELECTOR.context,
                  cl.mem_flags.READ_WRITE
                  | cl.mem_flags.ALLOC_HOST_PTR,
                  size=picture.size), picture, CURRENT_SELECTOR.queue)
    for picture in input_pictures_array
]

timer_list = []
queue_timer = []
# Create the OpenCL program and kernel
prgs = [
    cl.Program(get_context_queue().context, kernel_code_grayscale).build()
    for i in range(CONTEXT_COUNT)
]
prgs_grayscale = [prg.grayscale for prg in prgs]
local_size = (1, 1)
reset_selector()
for i in range(10):
    start_time = timeit.default_timer()
    queue_temp = []
    for opencl_buffer in opencl_input_buffers:
        get_context_queue()
        global_size = (math.ceil(opencl_buffer.width / CHUNK_PROCESSING_SIZE),
                       math.ceil(opencl_buffer.height))
        prgs_grayscale[CURRENT_SELECTOR.device_index].set_args(
            opencl_buffer.opencl_buffer_input,
            opencl_buffer.opencl_buffer_output, np.uint32(opencl_buffer.width),
            np.uint64(opencl_buffer.size), np.uint64(CHUNK_PROCESSING_SIZE))
        start_queue = timeit.default_timer()
        cl.enqueue_nd_range_kernel(
            CURRENT_SELECTOR.queue,
            prgs_grayscale[CURRENT_SELECTOR.device_index], global_size,
            local_size)
        end_queue = timeit.default_timer()
        queue_temp.append(end_queue - start_queue)
        # cl.enqueue_copy(CURRENT_SELECTOR.queue,
        #                 opencl_buffer.opencl_buffer_output,
        #                 opencl_buffer.opencl_buffer_input)

    for i in range(CONTEXT_COUNT):
        get_context_queue().queue.finish()
    end_time = timeit.default_timer()
    timer_list.append(end_time - start_time)
    queue_timer.append(np.average(max(queue_temp)))
    queue_temp = []

print("All values:")
print(" | ", end="")
for i in timer_list:
    print("{:.3f}".format(i), end=" | ")
print("\n")
print("max: " + str(max(timer_list)))
print("min:" + str(min(timer_list)))
print("mean: " + str(np.average(timer_list)))
print("Queue times: ")
for i in queue_timer:
    print("{:.3f}".format(i), end=" | ")
print("Queue time max: " + str(max(queue_timer)))
print("Queue time min: " + str(min(queue_timer)))
for filtered_picture in opencl_input_buffers:
    filtered_picture.save_output_on_disk(OUTPUT_PATH)

# Save the filtered image to disk

print("done")