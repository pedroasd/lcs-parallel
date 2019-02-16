/* pi.c
 gcc pi.c -o pi -lOpenCL -I.
 http://pastebin.com/gq3h0KVn
*/

#include <stdio.h>
#include <stdlib.h>
#include <err_code.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <sys/types.h>

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#define NUMTHREADS  128
#define WORKGROUPS  2
#define ITERATIONS  2e09


int main()
{

  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem d_pi = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  int          err;               // error code returned from OpenCL calls
  char string[MEM_SIZE];
  double h_pi[NUMTHREADS];
  int iterations;
  /******************************************************************************/
  /* open kernel */
  FILE *fp;
  char fileName[] = "./pi.cl";
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
  d_pi = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(double), NULL, &ret);
  checkError(ret, "Creating buffer d_pì");

  //print kernel
  //printf("\n%s\n%i bytes\n", source_str, (int)source_size); fflush(stdout);
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
  kernel = clCreateKernel(program, "calculatePi", &ret);
  checkError(ret, "Creating kernel");

  /* Set OpenCL Kernel Parameters */
  iterations = ITERATIONS;
  ret = clSetKernelArg(kernel, 0, sizeof(double), (void *)&d_pi);
  checkError(ret, "Setting kernel arguments");
  ret = clSetKernelArg(kernel, 1, sizeof(int), &iterations);
  checkError(ret, "Setting kernel arguments");

  //clEnqueueWriteBuffer(command_queue, pi, CL_TRUE, 0, 1, &h_pi, 0, NULL, NULL);
  size_t global_work_size = NUMTHREADS;
  size_t local_work_size = NUMTHREADS/WORKGROUPS;
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
  ret = clEnqueueReadBuffer(command_queue, d_pi, CL_TRUE, 0, NUMTHREADS * sizeof(double), h_pi, 0, NULL, NULL);
  checkError(ret, "Creating program");

  int i;
  for(i = 1; i < NUMTHREADS; i++)
    *h_pi = *h_pi + *(h_pi + i);
  printf("\n%1.12f", *h_pi);

  /* Finalization */
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(d_pi);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(source_str);

  return 0;
}
