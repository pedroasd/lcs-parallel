/**
 * Implementación del algoritmo de (Jiaoyun Yang, 2010)
 * http://www.iaeng.org/publication/WCE2010/WCE2010_pp499-504.pdf 
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define TAM_MAX 4.5e4

void lcs(char *a, char *b, int m, int n);
void lcs_openmp(char *a, char *b, int m, int n, int thread_count);
int max(int a, int b);
char *rand_string_alloc(size_t size);
void rand_string(char *str, size_t size);
char *alfabetoCadenas(char *alfab, char *a, int n);
char *adicionarChar(char *str, char caracter);
int buscarIndice(char *cadena, char a);

int main(void)
{
    int thread_count = omp_get_max_threads();
    for(int thread_count= 1 ; thread_count <=32 ; thread_count = thread_count * 2 ){
        for (int i = 2; i <= TAM_MAX;){
            double begin = omp_get_wtime();
            //char *a = "ABMDEBMA";
            //char *b = "ABACAEMC";
            char *a = rand_string_alloc(i);
            char *b = rand_string_alloc(i);
            int m = strlen(a);
            int n = strlen(b);
            //lcs_openmp(a, b, m, n, thread_count);
            lcs(a, b, m, n);
            free(a);
            free(b);

            double end = omp_get_wtime();
            double time_spent = end - begin;
            printf("N;%d;I;%d;T;%f\n", thread_count, i, time_spent);
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
/*
    // Comparación entre los dos algoritmos
        int i = 40000;


        //char *a = "ABMDEBMA";
        //char *b = "ABACAEMC";
        char *a = rand_string_alloc(i);
        char *b = rand_string_alloc(i);
        int m = strlen(a);
        int n = strlen(b);
        double begin = omp_get_wtime();
        lcs(a, b, m, n);
        double end = omp_get_wtime();
        double time_spent = end - begin;
        printf("I;%d;T;%f\n", i, time_spent);

        begin = omp_get_wtime();
        lcs_openmp(a, b, m, n, thread_count);
        end = omp_get_wtime();
        time_spent = end - begin;
        printf("I;%d;T;%f\n", i, time_spent);

        free(a);
        free(b);
*/
        
        
    return 0;
}

/** Función para obtener el máximo entre dos números.*/
int max(int a, int b)
{
    return (a > b) ? a : b;
}

/** Función que calcula una subsecuencia común más larga
 * (Longest Common Subsequence, LCS) entre dos cadenas de
 * caracteres: a de tamaño m y b de tamaño n, y la imprime.
*/
void lcs(char *a, char *b, int m, int n)
{
    // Alfabeto de las dos cadenas
    char *alfabeto = "";
    alfabeto = alfabetoCadenas(alfabeto, a, m);
    alfabeto = alfabetoCadenas(alfabeto, b, n);
    int l = strlen(alfabeto);

    //printf("Alfabeto: %s\n", alfabeto);

    // Tabla de preprocesamiento
    int *mpre;
    mpre = (int *)malloc(l * (n + 1) * sizeof(int));
    for (int i = 1; i <= l; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            //printf("%d,%d",i,j);
            if (j == 0)
                *(mpre + (i - 1) * (n + 1) + j) = 0;
            else if (*(b + j - 1) == alfabeto[i - 1])
                *(mpre + (i - 1) * (n + 1) + j) = j;
            else
                *(mpre + (i - 1) * (n + 1) + j) = *(mpre + (i - 1) * (n + 1) + j - 1);
        }
    }
    //printf("Preprocesamiento finalizado\n");

    // Impresión de tabla preprocesamiento.
    /*int g = 0;
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            printf("%d\t", *(mpre + g));
            g++;
        }
        printf("\n");
    }
    printf("\n");
    */

    // Matriz de resultado
    int *mres;
    mres = (int *)malloc((m + 1) * (n + 1) * sizeof(int));

    for (int i = 0; i <= m; i++)
    {
        int indiceAlfabeto = buscarIndice(alfabeto, *(a + i - 1));
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

    //printf("Programación dinámica\n");

    // Impresión de tabla resultado de programación dinámica.
    /*int k = 0;
    for (int i = 0; i <= m; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            printf("%d\t", *(mres + k));
            k++;
        }
        printf("\n");
    }*/

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
    //printf("Subsecuencia común más larga entre:\n C1: %s\n y C2: %s\n es %s\n", a, b, lcs);

    printf("Tamaño: %ld\n", strlen(lcs));
    free(mres);
    free(mpre);
}

/** Función que calcula una subsecuencia común más larga
 * (Longest Common Subsequence, LCS) entre dos cadenas de
 * caracteres: a de tamaño m y b de tamaño n, y la imprime.
*/
void lcs_openmp(char *a, char *b, int m, int n, int thread_count)
{
    // Alfabeto de las dos cadenas
    char *alfabeto = "";
    alfabeto = alfabetoCadenas(alfabeto, a, m);
    alfabeto = alfabetoCadenas(alfabeto, b, n);
    int l = strlen(alfabeto);

    //printf("Alfabeto: %s\n", alfabeto);

    // Tabla de preprocesamiento
    int *mpre;
    mpre = (int *)malloc(l * (n + 1) * sizeof(int));

    #pragma omp parallel for num_threads(thread_count)
    for (int i = 1; i <= l; i++)
    {
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

    
    //printf("Preprocesamiento finalizado\n");

    // Impresión de tabla preprocesamiento.
    /*int g = 0;
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            printf("%d\t", *(mpre + g));
            g++;
        }
        printf("\n");
    }
    printf("\n");
    */

    // Matriz de resultado
    int *mres;
    mres = (int *)malloc((m + 1) * (n + 1) * sizeof(int));

    for (int i = 0; i <= m; i++)
    {
        int indiceAlfabeto = buscarIndice(alfabeto, *(a + i - 1));

        #pragma omp parallel for num_threads(thread_count)
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

    //printf("Programación dinámica\n");

    // Impresión de tabla resultado de programación dinámica.
    /*int k = 0;
    for (int i = 0; i <= m; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            printf("%d\t", *(mres + k));
            k++;
        }
        printf("\n");
    }*/

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
    //printf("Subsecuencia común más larga entre:\n C1: %s\n y C2: %s\n es %s\n", a, b, lcs);

    //printf("Tamaño: %ld\n", strlen(lcs));
    free(mres);
    free(mpre);
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