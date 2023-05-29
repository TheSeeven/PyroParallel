class Kernels:

    GRAYSCALE = """
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

    EDGE_DETECTION = "TODO"
    DOUBLE_PRECISION = "TODO"

    def _GRAYSCALE():
        return Kernels.GRAYSCALE

    def _EDGE_DETECTION():
        return Kernels.EDGE_DETECTION

    def _DOUBLE_PRECISION():
        return Kernels.DOUBLE_PRECISION