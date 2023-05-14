import numpy as np
from PIL import Image
import pyopencl as cl
import os, psutil
import timeit
import math
import py3nvml.py3nvml as nvml

CHUNK_PROCESSING_SIZE = 32768
pid = os.getpid()
py_process = psutil.Process(pid)
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)


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
                        uint width,
                        ulong size,
                        const ulong chunk_size)
{
    width = chunk_size * width;
    ulong gid_column = get_global_id(0);
    ulong gid_row = get_global_id(1);
    uint chunks_per_column = width / chunk_size;
    ulong offset;
    float luma,r,g,b;
    offset = ((chunks_per_column*chunk_size*gid_row)+(chunk_size*gid_column))*3;
    for(ulong i = 0; i<(chunk_size*3) ; i+=3) {
        if(offset<size-3)
        {
            r = input[offset + 0 + i];
            g = input[offset + 1 + i];
            b = input[offset + 2 + i];

            luma = 0.299f * b + 0.587f * g + 0.114f * r;

            input[offset + 0 + i ] = luma;
            input[offset + 1 + i ] = luma;
            input[offset + 2 + i ] = luma; 
        }  
        else break;
    }
}
"""
mem_usage()
Image.MAX_IMAGE_PIXELS = None
picture_path = """C:\\Users\\peria\\Videos\\Mass Effect Andromeda\\25600x14400_1_NO_HDR.bmp"""
im = Image.open(picture_path)
im_arr = np.array(im)
mem_usage()
# Create Ope nCL context and command queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Create a buffer object in OpenCL with the numpy array data
im_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
                   im_arr.nbytes)
mem_usage()
cl.enqueue_copy(queue, im_buf, im_arr)
cl.enqueue_barrier(queue)
mem_usage()
# Create the OpenCL program and kernel
prg = cl.Program(ctx, kernel_code_grayscale).build()
grayscale_kernel = prg.grayscale
global_size = get_global_size(im_arr.shape[1], im_arr.shape[0],
                              CHUNK_PROCESSING_SIZE)
local_size = (1, 1)
grayscale_kernel.set_args(im_buf, np.uint32(global_size[0]),
                          np.uint64(im_buf.size),
                          np.uint64(CHUNK_PROCESSING_SIZE))

start_time = timeit.default_timer()
for i in range(10):
    cl.enqueue_nd_range_kernel(queue, grayscale_kernel, global_size,
                               local_size)

queue.finish()
end_time = timeit.default_timer()

total_time = end_time - start_time
print("total time: ", total_time)
im_filtered_arr = np.empty_like(im_arr)
cl.enqueue_copy(queue, im_filtered_arr, im_buf)
# Save the filtered image to disk
im_filtered_arr = Image.fromarray(im_filtered_arr)
im_filtered_arr.save("grayscale.bmp")

print("done")