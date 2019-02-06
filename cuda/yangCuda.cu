#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define BLOCKS  1024//512
#define NUMTHREADS 30000//8192
#define TAM 4.5e3

void lcs_cuda(char *a, char *b, int m, int n, int block_count, int thread_count);
char *alfabetoCadenas(char *alfab, char *a, int n);
char *adicionarChar(char *str, char caracter);
int buscarIndice(char *cadena, char a);
char *rand_string_alloc(size_t size);
void rand_string(char *str, size_t size);

/**
 * CUDA Kernel Device code
 * 
 */ 
/*****************************************************************************/

__global__ void preprocesamiento(int *mpre, char *b, char *alfabeto, long int n, int l)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x + 1;
    for (int j = 0; j <= n; j++)
    {
        //printf("%d,%d\n",i,j);
        if (j == 0)
            *(mpre + (i - 1) * (n + 1) + j) = 0;
        else if (*(b + j - 1) == alfabeto[i - 1])
            *(mpre + (i - 1) * (n + 1) + j) = j;
        else
            *(mpre + (i - 1) * (n + 1) + j) = *(mpre + (i - 1) * (n + 1) + j - 1);
    }
}

__global__ void matrizResultado(int *mpre, int *mres, int indiceAlfabeto, long int i, long int n, int threads)
{
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    int inicial = (n/threads) * index;
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
}
/******************************************************************************
 * Host main routine
 */
int main(int argc, char *argv[])
{   
    for(int block_count = 1; block_count <= BLOCKS; block_count = block_count * 2){
        for(int thread_count= 1 ; thread_count <= NUMTHREADS; thread_count = thread_count * 2 ){
            for (int i = 2; i <= TAM;){
                double begin = omp_get_wtime();
                //char *a = "ABMDEBMA";
                //char *b = "ABACAEMC";
                char *a = rand_string_alloc(TAM);
                char *b = rand_string_alloc(TAM);
                int m = strlen(a);
                int n = strlen(b);
                lcs_cuda(a, b, m, n, block_count, thread_count);
                //free(a);
                //free(b);
                double end = omp_get_wtime();
                double time_spent = end - begin;
                
                printf("B;%d;N;%d;I;%d;T;%f\n", block_count,thread_count, i, time_spent);
                if (i > 2048)
                    i += 5000;
                else
                {
                    i = i * 2;
                    if (i > 2048)
                        i = 5000;
                }
            }
        }
    }
    return 0;
}

/** Función que calcula una subsecuencia común más larga
 * (Longest Common Subsequence, LCS) entre dos cadenas de
 * caracteres: a de tamaño m y b de tamaño n, y la imprime.
*/
void lcs_cuda(char *a, char *b, int m, int n, int block_count, int thread_count)
{
    // Alfabeto de las dos cadenas
    char *alfabeto = "";
    alfabeto = alfabetoCadenas(alfabeto, a, m);
    alfabeto = alfabetoCadenas(alfabeto, b, n);
    int l = strlen(alfabeto);

    //printf("Alfabeto: %s\n", alfabeto);

    // Tabla de preprocesamiento
    int *mpre, *d_mpre;
    int sz_mpre = l * (n + 1) * sizeof(int);
    mpre = (int *)malloc(sz_mpre);
    
    cudaError_t err = cudaSuccess;

    char *d_b, *d_alfabeto;

    err = cudaMalloc((void **)&d_mpre, sz_mpre);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_b, sizeof(char)*strlen(b));
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_alfabeto, sizeof(char)*strlen(alfabeto));
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_b, b, sizeof(char)*strlen(b), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector b -> d_b from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_alfabeto, alfabeto, sizeof(char)*strlen(alfabeto), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector alf -> d_alf from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Preprocesamiento    
    preprocesamiento<<< 1, l >>>(d_mpre, d_b, d_alfabeto, n, l);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch preprocesamiento kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(mpre, d_mpre, sz_mpre, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector d_mpre->mpre from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //printf("Preprocesamiento finalizado\n");

    err = cudaFree(d_b);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_mpre (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_alfabeto);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_alfabeto (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    // Matriz de resultado
    int *mres, *d_mres;
    int sz_mres = (m + 1) * (n + 1) * sizeof(int);
    mres = (int *)malloc(sz_mres);

    err = cudaMalloc((void **)&d_mres, sz_mres);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_mres (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i <= m; i++)
    {
        int indiceAlfabeto = buscarIndice(alfabeto, *(a + i - 1));


        int threadsPerBlock = thread_count/block_count;
        int threads = block_count * threadsPerBlock;
        matrizResultado<<< BLOCKS,threadsPerBlock >>>(d_mpre, d_mres, indiceAlfabeto, i, n, threads);
        err = cudaGetLastError();
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to launch preprocesamiento kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }        
    }

    err = cudaMemcpy(mres, d_mres, sz_mres, cudaMemcpyDeviceToHost);
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
    
    err = cudaDeviceReset();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   
}

char *adicionarChar(char *str, char caracter)
{
    size_t len = strlen(str);
    char *str2 = new char[len + 2];
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
    char *s = new char[size + 1];
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