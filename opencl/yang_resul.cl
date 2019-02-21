#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void matrizResultado(__global int *mpre, __global int *mres, const int indiceAlfabeto, const int i, const int n, const int threads)
{
    int index = get_global_id(0);
    //printf("%d:",index);
    
    int inicial = (n/threads) * index;
    int final = inicial + (n/threads) - 1;
    //if(i > 0) printf("i: %d index: %d inicial: %d final: %d n: %d threads: %d\n", i, index, inicial,final,n,threads);
    
    if(final > n || final < 0){ final = n;}

    for (int j = inicial; j <= final; j++)
    {
        if (i == 0 || j == 0)
            *(mres + i * (n + 1) + j) = 0;
        else if (*(mpre + indiceAlfabeto * (n + 1) + j) == 0)
            *(mres + i * (n + 1) + j) = max(*(mres + (i - 1) * (n + 1) + j), 0);
        else
            *(mres + i * (n + 1) + j) = max(*(mres + (i - 1) * (n + 1) + j), *(mres + (i - 1) * (n + 1) + *(mpre + indiceAlfabeto * (n + 1) + j) - 1) + 1);
        if(i == 1) printf("%d",*(mres + i * (n + 1) + j));
    }    
}
