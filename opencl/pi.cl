/* pi.cl */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void calculatePi(__global double *piT, const int iterations)
{   int start, end;
    double piLocal = 0.0;
    int id = get_global_id(0);
    int numthreads = get_global_size(0);
    start = (iterations/numthreads) * id;
    end = (iterations/numthreads) * (id + 1);
    int i = start;

    do{
        piLocal = piLocal + (double)(4.0 / ((i*2)+1));
        i++;
        piLocal = piLocal - (double)(4.0 / ((i*2)+1));
        i++;
    }while(i < end);
    *(piT + id) = piLocal;
}
