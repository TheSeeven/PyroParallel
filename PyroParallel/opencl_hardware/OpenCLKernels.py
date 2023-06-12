class Kernels:

    GRAYSCALE = """
    __kernel void grayscale(__global const uchar *input,
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
            if(pixel<size-2){
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

    EDGE_DETECTION = """TODO"""

    OPERATION_FP64 = """
    __kernel void operation(__global const double* A,
                           __global const double* B,
                           __global double* C,
                           const ulong length,
                           const ulong chunk_size,
                           const int operationType)
    {
        ulong index = get_global_id(0);
        ulong remainder = get_global_id(1);
        ulong startIndex = index * chunk_size;
        ulong currentIndex;
        for (ulong j = 0; j < chunk_size; j++) {
            currentIndex = (startIndex + j)+(remainder*length);
            
            if (currentIndex < length) {
                if (operationType == 0) {
                    C[currentIndex] = A[currentIndex] / B[currentIndex];
                }
                else if (operationType == 1) {
                    C[currentIndex] = A[currentIndex] * B[currentIndex];
                }
                else if (operationType == 2) {
                    C[currentIndex] = A[currentIndex] + B[currentIndex];
                }
                else if (operationType == 3) {
                    C[currentIndex] = A[currentIndex] - B[currentIndex];
                }
            }
        }
    }
    """

    OPERATION_FP32 = """ 
    __kernel void operation(__global const float* A,
                           __global const float* B,
                           __global float* C,
                           const ulong length,
                           const ulong chunk_size,
                           const int operationType)
    {
        ulong index = get_global_id(0);
        ulong remainder = get_global_id(1);
        ulong startIndex = index * chunk_size;
        ulong currentIndex;
        for (ulong j = 0; j < chunk_size; j++) {
            currentIndex = (startIndex + j)+(remainder*length);
            
            if (currentIndex < length) {
                if (operationType == 0) {
                    C[currentIndex] = A[currentIndex] / B[currentIndex];
                }
                else if (operationType == 1) {
                    C[currentIndex] = A[currentIndex] * B[currentIndex];
                }
                else if (operationType == 2) {
                    C[currentIndex] = A[currentIndex] + B[currentIndex];
                }
                else if (operationType == 3) {
                    C[currentIndex] = A[currentIndex] - B[currentIndex];
                }
            }
        }
    }
    """

    @staticmethod
    def _GRAYSCALE():
        return Kernels.GRAYSCALE

    @staticmethod
    def _EDGE_DETECTION():
        return Kernels.EDGE_DETECTION

    @staticmethod
    def _OPERATION_FP32():
        return Kernels.OPERATION_FP32

    @staticmethod
    def _OPERATION_FP64():
        return Kernels.OPERATION_FP64