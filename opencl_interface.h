#ifndef OPENCL_INTERFACE
#define OPENCL_INTERFACE

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <vector>
#include <string>
#include <CL/opencl.hpp>

class OpenCLInterface
{
    public:
        bool isInitialized = false;
        OpenCLInterface();
        void initialize(const char *source, const char *programName,
                        size_t globalSize, std::vector<float*> inputPtrs,
                        std::vector<size_t>inputSizes, float* output,
                        size_t outputSize);
        void setGlobalWorkSize(size_t size);
        void setSource(const char* source, const char* name);
        void updateBuffer(const int bufferIndex, const float *newData);
        void setFloatArg(const int index, float value);
        void printInfo();
        void cleanup();
        void executeAndRead();
        void execute();
        void readResult();

    private:
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;
        const char* programSource = "No program";
        const char* programName = "No program name";
        std::vector<cl_mem> inBuffers;
        cl_mem outBuffer;
        size_t globalWorkSize;
        std::vector<size_t> inputSizes;
        size_t outputSize;
        float *output = nullptr;


        std::string getCodeExplanation(cl_int code);
        void printCodeExplanation(cl_int code);
        void getPlatformIDs();
        void getDeviceIDs();
        void createContext();
        void createCommandQueue();
        void createInBuffer(size_t bufferSize, float *data);
        void createOutBuffer(size_t bufferSize, float *data);
        void createProgram();
        int buildProgram();
        void createKernel();
        void setKernelArgs();
        void setKernelArg(int bufferIndex, float* newValues);
};

#endif // OPENCL_INTERFACE
