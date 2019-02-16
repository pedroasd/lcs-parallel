/* gcc hello.c -o hello -lOpenCL -I. */
/* http://pastebin.com/FZHyZmMk */

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/types.h>
#include "err_code.h"

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

int main()
{
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem op, memobj = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  int          err;               // error code returned from OpenCL calls

  char string[MEM_SIZE];

  FILE *fp;
  char fileName[] = "./hello.cl";
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

  /* Get Platform and Device Info */
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  /* Create OpenCL context */
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  /* Create Command Queue */
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  /* Create Memory Buffer */
  memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char), NULL, &ret);
  op = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char), NULL, &ret);

  printf("\n%s\n%i bytes\n", source_str, source_size); fflush(stdout);
  /* Create Kernel Program from the source */
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
  (const size_t *)&source_size, &ret);

  /* Build Kernel Program */
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      size_t len;
      char buffer[2048];
      printf("\nError: Failed to build program executable!\n%s\n", err_code(err));
  }
  /* Create OpenCL Kernel */
  kernel = clCreateKernel(program, "hello", &ret);

  /* Set OpenCL Kernel Parameters */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&op);

  char o = 'B';
  clEnqueueWriteBuffer(command_queue, op, CL_TRUE, 0, 1, &o, 0, NULL, NULL);

  /* Execute OpenCL Kernel */
  ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);

  /* Copy results from the memory buffer */
  ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
  MEM_SIZE * sizeof(char),string, 0, NULL, NULL);

  /* Display Result */
  puts(string);

  /* Finalization */
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(source_str);

  return 0;
}
