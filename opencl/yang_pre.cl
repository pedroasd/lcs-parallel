#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void preprocesamiento(__global int *mpre, __global char *b/*, __global char *alfabeto, const long int n, const int l*/)
{
    int i = get_global_id(0);

    printf("%c\n", &alfabeto[i]);

    /*for (int j = 0; j <= n; j++)
    {
        //printf("%d,%d\n",i,j);
        if (j == 0)
            *(mpre + (i - 1) * (n + 1) + j) = 0;
        else if (*(b + j - 1) == alfabeto[i - 1])
            *(mpre + (i - 1) * (n + 1) + j) = j;
        else
            *(mpre + (i - 1) * (n + 1) + j) = *(mpre + (i - 1) * (n + 1) + j - 1);
    }
    */
}