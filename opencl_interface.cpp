#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <vector>
#include <string>

#include "opencl_interface.h"

OpenCLInterface::OpenCLInterface(){
    this->getPlatformIDs();
    this->getDeviceIDs();
    this->createContext();
    this->createCommandQueue();
    std:: cout << "Construction done!\n";
}

void OpenCLInterface::initialize(const char* source, const char* programName,
                                 size_t globalSize, std::vector<float*> inputPtrs,
                                 std::vector<size_t>inputSizes, float* output,
                                 size_t outputSize){
    this->output = output;
    this->outputSize = outputSize;
    this->globalWorkSize = globalSize;
    this->inputSizes = inputSizes;
    for (int i = 0 ; i < inputPtrs.size() ; i++){
        this->createInBuffer(inputSizes.at(i), inputPtrs.at(i));
    }
    this->createOutBuffer(outputSize, output);

    this->setSource(source, programName);
    this->createProgram();
    this->buildProgram();
    this->createKernel();
    this->setKernelArgs();
    std::cout << "Interface initialized successfully!\n\n";
    this->isInitialized = true;
}

std::string OpenCLInterface::getCodeExplanation(cl_int code){
    std::string codeExplanation;
    switch (code){
        case 0:
            codeExplanation = "CL_SUCCESS";
            break;
        case -1:
            codeExplanation = "CL_DEVICE_NOT_FOUND";
            break;
        case -2:
            codeExplanation = "CL_DEVICE_NOT_AVAILABLE";
            break;
        case -3:
            codeExplanation = "CL_COMPILER_NOT_AVAILABLE";
            break;
        case -4:
            codeExplanation = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            break;
        case -5:
            codeExplanation = "CL_OUT_OF_RESOURCES";
            break;
        case -6:
            codeExplanation = "CL_OUT_OF_HOST_MEMORY";
            break;
        case -7:
            codeExplanation = "CL_PROFILING_INFO_NOT_AVAILABLE";
            break;
        case -8:
            codeExplanation = "CL_MEM_COPY_OVERLAP";
            break;
        case -9:
            codeExplanation = "CL_IMAGE_FORMAT_MISMATCH";
            break;
        case -10:
            codeExplanation = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            break;
        case -11:
            codeExplanation = "CL_BUILD_PROGRAM_FAILURE";
            break;
        case -12:
            codeExplanation = "CL_MAP_FAILURE";
            break;
        case -13:
            codeExplanation = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            break;
        case -14:
            codeExplanation = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            break;
        case -15:
            codeExplanation = "CL_COMPILE_PROGRAM_FAILURE";
            break;
        case -16:
            codeExplanation = "CL_LINKER_NOT_AVAILABLE";
            break;
        case -17:
            codeExplanation = "CL_LINK_PROGRAM_FAILURE";
            break;
        case -18:
            codeExplanation = "CL_DEVICE_PARTITION_FAILED";
            break;
        case -19:
            codeExplanation = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            break;
        case -30:
            codeExplanation = "CL_INVALID_VALUE";
            break;
        case -31:
            codeExplanation = "CL_INVALID_DEVICE_TYPE";
            break;
        case -32:
            codeExplanation = "CL_INVALID_PLATFORM";
            break;
        case -33:
            codeExplanation = "CL_INVALID_DEVICE";
            break;
        case -34:
            codeExplanation = "CL_INVALID_CONTEXT";
            break;
        case -35:
            codeExplanation = "CL_INVALID_QUEUE_PROPERTIES";
            break;
        case -36:
            codeExplanation = "CL_INVALID_COMMAND_QUEUE";
            break;
        case -37:
            codeExplanation = "CL_INVALID_HOST_PTR";
            break;
        case -38:
            codeExplanation = "CL_INVALID_MEM_OBJECT";
            break;
        case -39:
            codeExplanation = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            break;
        case -40:
            codeExplanation = "CL_INVALID_IMAGE_SIZE";
            break;
        case -41:
            codeExplanation = "CL_INVALID_SAMPLER";
            break;
        case -42:
            codeExplanation = "CL_INVALID_BINARY";
            break;
        case -43:
            codeExplanation = "CL_INVALID_BUILD_OPTIONS";
            break;
        case -44:
            codeExplanation = "CL_INVALID_PROGRAM";
            break;
        case -45:
            codeExplanation = "CL_INVALID_PROGRAM_EXECUTABLE";
            break;
        case -46:
            codeExplanation = "CL_INVALID_KERNEL_NAME";
            break;
        case -47:
            codeExplanation = "CL_INVALID_KERNEL_DEFINITION";
            break;
        case -48:
            codeExplanation = "CL_INVALID_KERNEL";
            break;
        case -49:
            codeExplanation = "CL_INVALID_ARG_INDEX";
            break;
        case -50:
            codeExplanation = "CL_INVALID_ARG_VALUE";
            break;
        case -51:
            codeExplanation = "CL_INVALID_ARG_SIZE";
            break;
        case -52:
            codeExplanation = "CL_INVALID_KERNEL_ARGS";
            break;
        case -53:
            codeExplanation = "CL_INVALID_WORK_DIMENSION";
            break;
        case -54:
            codeExplanation = "CL_INVALID_WORK_GROUP_SIZE";
            break;
        case -55:
            codeExplanation = "CL_INVALID_WORK_ITEM_SIZE";
            break;
        case -56:
            codeExplanation = "CL_INVALID_GLOBAL_OFFSET";
            break;
        case -57:
            codeExplanation = "CL_INVALID_EVENT_WAIT_LIST";
            break;
        case -58:
            codeExplanation = "CL_INVALID_EVENT";
            break;
        case -59:
            codeExplanation = "CL_INVALID_OPERATION";
            break;
        case -60:
            codeExplanation = "CL_INVALID_GL_OBJECT";
            break;
        case -61:
            codeExplanation = "CL_INVALID_BUFFER_SIZE";
            break;
        case -62:
            codeExplanation = "CL_INVALID_MIP_LEVEL";
            break;
        case -63:
            codeExplanation = "CL_INVALID_GLOBAL_WORK_SIZE";
            break;
        case -64:
            codeExplanation = "CL_INVALID_PROPERTY";
            break;
        case -65:
            codeExplanation = "CL_INVALID_IMAGE_DESCRIPTOR";
            break;
        case -66:
            codeExplanation = "CL_INVALID_COMPILER_OPTIONS";
            break;
        case -67:
            codeExplanation = "CL_INVALID_LINKER_OPTIONS";
            break;
        case -68:
            codeExplanation = "CL_INVALID_DEVICE_PARTITION_COUNT";
        default:
            std::cout << "Code not found!\n";
        }
    return codeExplanation;
}

void OpenCLInterface::printCodeExplanation(cl_int code){
    std::string explanation = this->getCodeExplanation(code);
    std::cout << explanation << std::endl;
}

void OpenCLInterface::printInfo(){
    std::cout << "Platform ID: " << this->platform << std::endl;
    std::cout << "Device ID: " << this->device << std::endl;
    if (!this->isInitialized){
        std::cout << "Initialized: false\n";
    } else {
        std::cout << "Program name: " << this->programName << std::endl;
        std::cout << "Program source: \n\n" << this->programSource << std::endl;
        std::cout << "Global work size: " << globalWorkSize << "\n";
        std::cout << "Input sizes: ";

        int nInBuffers = this->inputSizes.size();
        size_t *inSizes = this->inputSizes.data();
        for (int i = 0 ; i < nInBuffers-1 ; i++){
            std::cout << inSizes[i] << ", ";
        }
        std::cout << inSizes[nInBuffers-1] << "\n";
        std::cout << "Output size: " << this->outputSize << "\n";
        std::cout << "Global work size: " << globalWorkSize << "\n";
    }


    std::cout << "\n\n";
}

void OpenCLInterface::getPlatformIDs(){
    cl_int result = clGetPlatformIDs(1, &(this->platform), NULL);
    std::cout << "Get platform IDs: " 
              << this->getCodeExplanation(result)
              << std::endl;
}

void OpenCLInterface::getDeviceIDs(){
    cl_int result = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU, 1, &(this->device), NULL);
    std::cout << "Get device IDs: " 
              << this->getCodeExplanation(result)
              << std::endl;
}

void OpenCLInterface::createContext(){
    cl_int result;
    this->context = clCreateContext(0, 1, &(this->device), NULL, NULL, &result);
    std::cout << "Create Context: " 
              << this->getCodeExplanation(result)
              << std::endl;
}

void OpenCLInterface::createCommandQueue(){
    cl_int result;
    this->queue = clCreateCommandQueueWithProperties(this->context, this->device, 0, &result);
}

void OpenCLInterface::createInBuffer(size_t bufferSize, float *data){
    cl_int result;
    cl_mem buffer = clCreateBuffer(this->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                      bufferSize, data, &result);
    std::cout << "Create buffer: " 
              << this->getCodeExplanation(result)
              << std::endl;
    this->inBuffers.push_back(buffer);
}

void OpenCLInterface::createOutBuffer(size_t bufferSize, float *data){
    cl_int result;
    cl_mem buffer = clCreateBuffer(this->context, CL_MEM_WRITE_ONLY, 
                                      bufferSize, NULL, &result);
    std::cout << "Create buffer: " 
              << this->getCodeExplanation(result)
              << std::endl;
    this->outBuffer = buffer;
}

void OpenCLInterface::setGlobalWorkSize(size_t size){
    this->globalWorkSize = size;
}

void OpenCLInterface::setSource(const char* source, const char* name){
    this->programSource = source;
    this->programName = name;
}

void OpenCLInterface::createProgram(){
    cl_int result;
    this->program = clCreateProgramWithSource(this->context, 1, &(this->programSource), NULL, &result);
    std::cout << "Create program: " 
              << this->getCodeExplanation(result)
              << std::endl;
}

int OpenCLInterface::buildProgram(){
    cl_int result = clBuildProgram(this->program, 1, &(this->device), NULL, NULL, NULL);
    std::cout << "Build program: " 
              << this->getCodeExplanation(result)
              << std::endl;
    if (result != CL_SUCCESS){
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL);
        fprintf(stderr, "OpenCL Build Error:\n%s\n", log.data());
        return -1;
    } else {
        return 0;
    }
}

void OpenCLInterface::createKernel(){
    cl_int result;
    this->kernel = clCreateKernel(program, this->programName, &result);
    std::cout << "Create kernel: " 
              << this->getCodeExplanation(result)
              << std::endl;
}

void OpenCLInterface::setKernelArgs(){
    int counter = 0;
    for (int i = 0 ; i < this->inBuffers.size() ; i++){
        clSetKernelArg(this->kernel, counter, sizeof(cl_mem), &this->inBuffers.at(i));
        counter++;
    }
    clSetKernelArg(this->kernel, counter, sizeof(cl_mem), &this->outBuffer);
    counter++;
}

void OpenCLInterface::updateBuffer(const int bufferIndex, const float* newData){
    size_t size = this->inputSizes.at(bufferIndex);
    cl_int result = clEnqueueWriteBuffer(
        this->queue,
        this->inBuffers.at(bufferIndex),  // Existing valid cl_mem handle
        CL_TRUE,                 // Blocking write
        0,                       // Offset
        size,                    // Size in bytes
        newData,                 // Host pointer with NEW data
        0, NULL, NULL
    );
    std::cout << "Update buffer " << bufferIndex << ": " 
              << this->getCodeExplanation(result)
              << std::endl;
}

void OpenCLInterface::setFloatArg(const int index, float value) {
    cl_int result = clSetKernelArg(this->kernel, index, sizeof(float), &value);
    std::cout << "Update float index " << index << ": " 
              << this->getCodeExplanation(result)
              << std::endl;
}

void OpenCLInterface::executeAndRead(){
    this->execute();
    this->readResult();
}

void OpenCLInterface::execute(){
    if (this->isInitialized){
        clEnqueueNDRangeKernel(this->queue, this->kernel, 1, NULL, &(this->globalWorkSize), NULL, 0, NULL, NULL);
        clFinish(queue);
    } else {
        std::cout << "Interface not initialized!\n";
    }
}

void OpenCLInterface::readResult(){
    if (this->isInitialized){
        clEnqueueReadBuffer(this->queue, this->outBuffer, CL_TRUE, 0,
                            this->outputSize, this->output, 0, NULL, NULL);
    } else {
        std::cout << "Interface not initialized!\n";
    }
}

void OpenCLInterface::cleanup(){
    clReleaseKernel(this->kernel);
    clReleaseProgram(this->program);

    for (int i = 0 ; i < this->inBuffers.size() ; i++){
        clReleaseMemObject(this->inBuffers.at(i));
    }
    clReleaseMemObject(this->outBuffer);

    clReleaseCommandQueue(this->queue);
    clReleaseContext(this->context);
}
