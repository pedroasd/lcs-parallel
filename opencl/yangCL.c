#include <stdio.h>
#include <stdlib.h>
#include <err_code.h>
#include <CL/cl.h>
#include <sys/types.h>
#include <string.h>
#include <omp.h>

#define MAX_SOURCE_SIZE (0x100000)
#define WORKGROUPS  2

#define BLOCKS  512
#define NUMTHREADS 8192 //128
#define TAM 1.5e4
#define MAX_THREADS_BLOCK 1024

void lcs_opencl(char *a, char *b, int m, int n, int block_count, int thread_count);
int max(int a, int b);
char *alfabetoCadenas(char *alfab, char *a, int n);
char *adicionarChar(char *str, char caracter);
int buscarIndice(char *cadena, char a);
char *rand_string_alloc(size_t size);
void rand_string(char *str, size_t size);

int main(int argc, char *argv[])
{   
    int i = TAM;
    int block_count = BLOCKS;
    int thread_count = NUMTHREADS;
    //int thread_count = TAM/2;
    //for(int block_count = 2; block_count <= BLOCKS; block_count=block_count+50){
        //for(int thread_count= block_count; thread_count <= NUMTHREADS; thread_count=thread_count+50 ){
            //for (int i = 2; i <= TAM;){

                int threadsPerBlock = thread_count/block_count;
                if(threadsPerBlock <= MAX_THREADS_BLOCK){

                    double begin = omp_get_wtime();
                    char *a = "ABMDEBMA";
                    char *b = "ABACAEMC";
                    //char *a = rand_string_alloc(i);
                    //char *b = rand_string_alloc(i);
                    int m = strlen(a);
                    int n = strlen(b);
                    
                    //printf("B;%d;N;%d;I;%d\n", block_count,thread_count, i);
                    lcs_opencl(a, b, m, n, block_count, thread_count);
                    
                    free(a);
                    free(b);
                    double end = omp_get_wtime();
                    double time_spent = end - begin;
                    
                    printf("B;%d;N;%d;I;%d;T;%f\n", block_count,thread_count, i, time_spent);
                }

                /*if (i > 2048)
                    i += 5000;
                else
                {
                    i = i * 2;
                    if (i > 2048)
                        i = 5000;
                }*/
                
            //}
        //}
    //}
    return 0;
}

/** Función que calcula una subsecuencia común más larga
 * (Longest Common Subsequence, LCS) entre dos cadenas de
 * caracteres: a de tamaño m y b de tamaño n, y la imprime.
*/
void lcs_opencl(char *a, char *b, int m, int n, int block_count, int thread_count)
{
    // Alfabeto de las dos cadenas
    char *alfabeto = "";
    alfabeto = alfabetoCadenas(alfabeto, a, m);
    alfabeto = alfabetoCadenas(alfabeto, b, n);
    int l = strlen(alfabeto);

    printf("Alfabeto: %s\n", alfabeto);

    // Tabla de preprocesamiento
    int *mpre;
    int sz_mpre = l * (n + 1) * sizeof(int);
    mpre = (int *)malloc(sz_mpre);
    
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem d_mpre = NULL;
    cl_mem d_b = NULL;
    cl_mem d_alfabeto = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    int          err;               // error code returned from OpenCL calls

    /******************************************************************************/
    /* open kernel */
    FILE *fp;
    char fileName[] = "./yang_pre.cl";
    char *source_str;
    size_t source_size;

    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);    

    /******************************************************************************/
    /* create objects */

    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    checkError(ret, "Creating context");

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    checkError(ret, "Creating queue");

    /* Create Memory Buffer */
    d_mpre = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_mpre, NULL, &ret);
    checkError(ret, "Creating buffer d_mpre");

    d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char)*strlen(b), NULL, &ret);
    checkError(ret, "Creating buffer d_b");

    d_alfabeto = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char)*strlen(alfabeto), NULL, &ret);
    checkError(ret, "Creating buffer d_alfabeto");

    /* copy array b and alfabeto to d_b and d_alfabeto */
    //clEnqueueWriteBuffer (command_queue, d_b, CL_FALSE, 0, sizeof(char)*strlen(b), b, 0, NULL, NULL);
    //clEnqueueWriteBuffer (command_queue, d_alfabeto, CL_FALSE, 0, sizeof(char)*strlen(alfabeto), alfabeto, 0, NULL, NULL);

    /******************************************************************************/
    /* create build program */

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
    (const size_t *)&source_size, &ret);
    checkError(ret, "Creating program");

    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(ret));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "preprocesamiento", &ret);
    checkError(ret, "Creating kernel");

    /* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg(kernel, 0, sizeof (cl_mem), (void *)&d_mpre);
    checkError(ret, "Setting kernel arguments");
    ret = clSetKernelArg(kernel, 1, sizeof (cl_mem), (void *)&d_b);
    checkError(ret, "Setting kernel arguments");
    ret = clSetKernelArg(kernel, 2, sizeof (cl_mem), (void *)&d_alfabeto);
    checkError(ret, "Setting kernel arguments");
    /*ret = clSetKernelArg(kernel, 3, sizeof(long), &n);
    checkError(ret, "Setting kernel arguments");
    ret = clSetKernelArg(kernel, 4, sizeof(int), &l);
    checkError(ret, "Setting kernel arguments");
    */

    //clEnqueueWriteBuffer(command_queue, pi, CL_TRUE, 0, 1, &h_pi, 0, NULL, NULL);
    size_t global_work_size = l;
    size_t local_work_size = 1;
    cl_uint work_dim = 1;
    /* Execute OpenCL Kernel */
    //ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);  //single work item
    ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
        0, &global_work_size, &local_work_size, 0, NULL, NULL);
    checkError(ret, "Enqueueing kernel");
    ret = clFinish(command_queue);
    checkError(ret, "Waiting for commands to finish");

    /******************************************************************************/
    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, d_mpre, CL_TRUE, 0, sz_mpre, mpre, 0, NULL, NULL);
    checkError(ret, "Creating program");
    
    //printf("Preprocesamiento finalizado\n");

    ret = clFlush(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(d_b);
    ret = clReleaseMemObject(d_alfabeto);

    
    // Matriz de resultado
    int *mres, *d_mres;
    int sz_mres = (m + 1) * (n + 1) * sizeof(int);
    mres = (int *)malloc(sz_mres);

    /*err = cudaMalloc((void **)&d_mres, sz_mres);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_mres (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    */

    int threadsPerBlock = thread_count/block_count;
    //int threadsPerBlock = (n/2)/block_count;
    int threads = block_count * threadsPerBlock;
    //printf("B;%d;N;%d;TB;%d\n", block_count,thread_count, threadsPerBlock);

    for (int i = 0; i <= m; i++)
    {
        int indiceAlfabeto = buscarIndice(alfabeto, *(a + i - 1));
        /*matrizResultado<<< block_count,threadsPerBlock >>>(d_mpre, d_mres, indiceAlfabeto, i, n, threads);
        err = cudaGetLastError();
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to launch matrizResultado kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        } 
        */



       for (int j = 0; j <= n; j++)
        {
            if (i == 0 || j == 0)
                *(mres + i * (n + 1) + j) = 0;
            else if (*(mpre + indiceAlfabeto * (n + 1) + j) == 0)
                *(mres + i * (n + 1) + j) = max(*(mres + (i - 1) * (n + 1) + j), 0);
            else
                *(mres + i * (n + 1) + j) = max(*(mres + (i - 1) * (n + 1) + j), *(mres + (i - 1) * (n + 1) + *(mpre + indiceAlfabeto * (n + 1) + j) - 1) + 1);
        }

    }

    /*err = cudaMemcpy(mres, d_mres, sz_mres, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector d_mres->mres from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_mpre);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_mpre (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_mres);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_mres (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    */

    //printf("Programación dinámica\n"); 

    // IMPRESIÓN DE LA CADENA
    // Longitud máxima de LCS
    int index = *(mres + m * (n + 1) + n);

    // Cadena donde se almacena una cadena LCS.
    char lcs[index + 1];
    lcs[index] = '\0';

    // Comienza desde la esquina inferior derecha y
    // almacena el resultado en lcs[]
    int i = m, j = n;
    while (i > 0 && j > 0)
    {
        // Si a[] y b[] son iguales, hace parte de LCS
        if (*(a + i - 1) == *(b + j - 1))
        {
            lcs[index - 1] = *(a + i - 1);
            i--;
            j--;
            index--;
        }

        // Si no es el mismo busca el mayor entre el superior y el izquierdo.
        else if (*(mres + (i - 1) * (n + 1) + j) > *(mres + i * (n + 1) + j - 1))
            i--;
        else
            j--;
    }

    // Imprime el resultado.
    printf("Subsecuencia común más larga entre:\n C1: %s\n y C2: %s\n es %s\n", a, b, lcs);

    
    
    //printf("Tamaño: %ld\n", strlen(lcs));
    free(mres);
    free(mpre);
    
    /*err = cudaDeviceReset();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 
    */  
}

/** Función para obtener el máximo entre dos números.*/
int max(int a, int b)
{
    return (a > b) ? a : b;
}

char *adicionarChar(char *str, char caracter)
{
    size_t len = strlen(str);
    char *str2 = malloc(len + 2);
    strcpy(str2, str);
    str2[len] = caracter;
    str2[len + 1] = '\0';
    return str2;
}

char *alfabetoCadenas(char *alfab, char *a, int n)
{
    char *alfabeto = alfab;
    for (int i = 0; i < n; i++)
    {
        int encontro = 0;
        for (int j = 0; j < strlen(alfabeto); j++)
        {
            if (*(a + i) == alfabeto[j])
            {
                encontro = 1;
                break;
            }
        }

        if (encontro == 0)
        {
            alfabeto = adicionarChar(alfabeto, a[i]);
        }
    }
    return alfabeto;
}

/** Retorna el índice donde se encuentra el caracter a en la cadena. */
int buscarIndice(char *cadena, char a)
{
    int tam = strlen(cadena);
    for (int j = 0; j < tam; j++)
        if (a == cadena[j])
            return j;
    return -1;
}

/** Genera una cadena de caracteres aleatoria de tamaño size y genera el esapcio en memoria. */
char *rand_string_alloc(size_t size)
{
    char *s = malloc(size + 1);
    if (s)
    {
        rand_string(s, size);
    }
    return s;
}

/** Genera una cadena de caracteres aleatoria en str de tamaño size. */
void rand_string(char *str, size_t size)
{
    const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    if (size)
    {
        --size;
        for (size_t n = 0; n < size; n++)
        {
            int key = rand() % (int)(sizeof charset - 1);
            *(str + n) = charset[key];
        }
        *(str + size) = '\0';
    }
}