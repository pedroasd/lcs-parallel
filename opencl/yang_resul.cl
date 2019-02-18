#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__global__ void matrizResultado(int *mpre, int *mres, int indiceAlfabeto, long int i, long int n, int threads)
{
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    printf("%d\n", index)

    /*int inicial = (n/threads) * index;
    int final = inicial + (n/threads) - 1;
    
    if(final > n || final < 0){ final = n;}

    for (int j = inicial; j <= final; j++)
    {
        if (i == 0 || j == 0)
            *(mres + i * (n + 1) + j) = 0;
        else if (*(mpre + indiceAlfabeto * (n + 1) + j) == 0)
            *(mres + i * (n + 1) + j) = max(*(mres + (i - 1) * (n + 1) + j), 0);
        else
            *(mres + i * (n + 1) + j) = max(*(mres + (i - 1) * (n + 1) + j), *(mres + (i - 1) * (n + 1) + *(mpre + indiceAlfabeto * (n + 1) + j) - 1) + 1);
    }
    */
}
