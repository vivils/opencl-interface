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

struct OpenCLImage {
    int index;
    bool isInput;
    cl_mem handle;
    cl_image_format format;
    cl_image_desc desc = {0};
    float *data;
};

class OpenCLInterface
{
    public:
        bool isInitialized;
        bool errorEncountered;
        OpenCLInterface();
        void initialize(const char* source,
                        const char* programName,
                        cl_uint workDimensions,
                        size_t *globalWorkSize,
                        std::vector<size_t> inputNumElements,
                        std::vector<float*> inputPtrs,
                        std::vector<size_t> outputNumElements,
                        std::vector<float*> outputPtrs);
        void setGlobalWorkSize(size_t *size);
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
        cl_uint workDimensions;
        size_t *globalWorkSize;
        size_t numArguments;
        std::vector<OpenCLBuffer> inBuffers = {};
        std::vector<OpenCLBuffer> outBuffers = {};
        std::vector<OpenCLImage> inImages = {};
        std::vector<OpenCLImage> outImages = {};


        std::string getCodeExplanation(cl_int code);
        void printCodeExplanation(cl_int code);
        int getPlatformIDs();
        int getDeviceIDs();
        int createContext();
        int createCommandQueue();
        void updateArgNum();
        int newBuffer(size_t numElements, float *data, bool isInput);
        int newImage(int width, int height, int depth,
                              float *data, bool isInput);
        int createBuffer(size_t bufferSize, float *data, cl_mem *handle, bool isInput);
        int createImage(cl_image_format *format, cl_image_desc *desc,
                                         float *data, cl_mem *outHandle, bool isInput);
        int createProgram();
        int buildProgram();
        int createKernel();
        int setAllKernelArgs();
        int setKernelArg(const int index, cl_mem handle);

};

#endif // OPENCL_INTERFACE
