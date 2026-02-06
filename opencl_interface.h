#ifndef OPENCL_INTERFACE
#define OPENCL_INTERFACE

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <CL/opencl.hpp>

struct OpenCLBuffer {
    size_t index;
    size_t numElements;
    size_t sizeBytes;
    bool isInput;
    float *data = nullptr;
    cl_mem handle = nullptr;
};

class OpenCLInterface
{
    public:
        bool isInitialized;
        bool errorEncountered;
        OpenCLInterface();
        void initialize(const char* source,
                        const char* programName,
                        size_t globalWorkSize,
                        std::vector<size_t> inputNumElements,
                        std::vector<float*> inputPtrs,
                        std::vector<size_t> outputNumElements,
                        std::vector<float*> outputPtrs);
        void setGlobalWorkSize(size_t size);
        void setSource(const char* source, const char* name);
        float* getBufferDataPtr(const int index, bool isInput);
        void updateBuffer(const int index);
        void printInfo();
        void cleanup();
        void executeAndRead(const int index);
        void execute();
        void readResult(const int index);

    private:
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;
        const char* programSource = "No program";
        const char* programName = "No program name";
        size_t globalWorkSize;
        std::vector<OpenCLBuffer> inBuffers = {};
        std::vector<OpenCLBuffer> outBuffers = {};

        std::string getCodeExplanation(cl_int code);
        void printCodeExplanation(cl_int code);
        int getPlatformIDs();
        int getDeviceIDs();
        int createContext();
        int createCommandQueue();
        int newBuffer(size_t numElements, float *data, bool isInput);
        int createInBuffer(size_t bufferSize, float *data, cl_mem *handle);
        int createOutBuffer(size_t bufferSize, float *data, cl_mem *handle);
        int createProgram();
        int buildProgram();
        int createKernel();
        int setAllKernelArgs();
        int setKernelArg(const int index, cl_mem handle);

};

#endif // OPENCL_INTERFACE
