#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <vector>
#include <string>

#include "opencl_interface.h"

OpenCLInterface::OpenCLInterface(){
    this->isInitialized = false;
    this->errorEncountered = false;
    try {
        if (this->getPlatformIDs() != 0){
            throw std::runtime_error("");
        }
        if (this->getDeviceIDs() != 0){
            throw std::runtime_error("");
        }
        if (this->createContext() != 0){
            throw std::runtime_error("");
        }
        if (this->createCommandQueue() != 0){
            throw std::runtime_error("");
        }

        std::cout << "Interface constructed successfully!\n\n";
        this->isInitialized = true;
    }
    catch (const std::exception& e){
        std::cerr << "Error: Couldn't construct OpenCL interface" << e.what() << std::endl;
        this->errorEncountered = true;
    }
}

void OpenCLInterface::initialize(const char* programName,
                                 const char* source,
                                 size_t globalWorkSize,
                                 std::vector<size_t> inputNumElements,
                                 std::vector<float*> inputPtrs,
                                 std::vector<size_t> outputNumElements,
                                 std::vector<float*> outputPtrs){
    try {
        if (inputNumElements.size() != inputPtrs.size()){
            throw std::runtime_error("Length of input data pointers and length of input sizes don't match!");
        }

        if (outputNumElements.size() != outputPtrs.size()){
            throw std::runtime_error("Length of output data pointers and length of output sizes don't match!");
        }

        this->globalWorkSize = globalWorkSize;

        int numElements;
        float *dataPtr;
        bool isInput;
        for (int i = 0 ; i < inputPtrs.size() ; i++){
            numElements = inputNumElements[i];
            dataPtr = inputPtrs.at(i);
            isInput = true;
            if(this->newBuffer(numElements, dataPtr, isInput) != 0){
                throw std::runtime_error("");
            }
        }
        int numOutputs = outputPtrs.size();
        std::cout << "Num outputs: " << numOutputs << std::endl;
        for (int i = 0 ; i < numOutputs ; i++){
            numElements = outputNumElements[i];
            dataPtr = outputPtrs[i];
            isInput = false;
            if(this->newBuffer(numElements, dataPtr, isInput) != 0){
                throw std::runtime_error("");
            }
        }

        this->setSource(source, programName);
        if (this->createProgram() != 0){
            throw std::runtime_error("");
        }
        if (this->buildProgram() != 0){
            throw std::runtime_error("");
        }
        if (this->createKernel() != 0){
            throw std::runtime_error("");
        }

        if (this->setAllKernelArgs() != 0){
            throw std::runtime_error("");
        }
        std::cout << "Interface initialized successfully!\n\n";
        this->isInitialized = true;
    } catch (const std::exception& e){
        std::cerr << "Error: Couldn't initialize OpenCL interface: " << e.what() << std::endl;
        this->errorEncountered = true;
    }
}

int OpenCLInterface::newBuffer(size_t numElements, float *data, bool isInput){
    int index = this->inBuffers.size() + this->outBuffers.size();
    size_t sizeBytes = numElements*sizeof(float);
    cl_mem handle = nullptr;

    OpenCLBuffer buffer;
    buffer.index = index;
    buffer.numElements = numElements;
    buffer.sizeBytes = sizeBytes;
    buffer.isInput = isInput;
    buffer.data = data;
    
    try {
        int result;
        if (buffer.isInput){
            result = this->createInBuffer(sizeBytes, data, &handle);
        } else {
            result = this->createOutBuffer(sizeBytes, data, &handle);
        }
        if (result != 0){
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Create buffer failed: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    buffer.handle = handle;
    if (isInput){
        this->inBuffers.push_back(buffer);
    } else {
        this->outBuffers.push_back(buffer);
    }

    if (handle == nullptr){
        std::cout << "handle is nullptr\n";
    }
    return 0;
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
        std::cout << "Buffers:\n";

        OpenCLBuffer *buffer;
        for (int i = 0 ; i < this->inBuffers.size() ; i++){
            buffer = &this->inBuffers.at(i);
            printf("    Index: %d, size: %d, direction: input\n",
                    buffer->index, buffer->numElements);
        }
        for (int i = 0 ; i < this->outBuffers.size() ; i++){
            buffer = &this->outBuffers.at(i);
            printf("    Index: %d, size: %d, direction: output\n",
                    buffer->index, buffer->numElements);
        }
    }
    std::cout << "\n\n";
}

int OpenCLInterface::getPlatformIDs(){
    try {
        cl_int result = clGetPlatformIDs(1, &(this->platform), NULL);
        if (result == CL_SUCCESS) {
            std::cout << "Platform ID received\n";
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't get platform ID: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

int OpenCLInterface::getDeviceIDs(){
    try {
        cl_int result = clGetDeviceIDs(this->platform,
                                       CL_DEVICE_TYPE_GPU,
                                       1,
                                       &(this->device),
                                       NULL);
        if (result == CL_SUCCESS) {
            std::cout << "Device ID received\n";
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't get device ID: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

int OpenCLInterface::createContext(){
    try {
    cl_int result;
        this->context = clCreateContext(0, 1, &(this->device), NULL, NULL, &result);
        if (result == CL_SUCCESS) {
            std::cout << "Context created\n";
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't create context: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

int OpenCLInterface::createCommandQueue(){
    try {
        cl_int result;
        this->queue = clCreateCommandQueueWithProperties(this->context, this->device, 0, &result);
        if (result == CL_SUCCESS) {
            std::cout << "Command queue created\n";
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't create command queue: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

int OpenCLInterface::createInBuffer(size_t bufferSize, float *data, cl_mem *outHandle){
    try {
        cl_int result;
        cl_mem handle = clCreateBuffer(this->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                          bufferSize, data, &result);
        if (result == CL_SUCCESS) {
            std::cout << "Input buffer created with size: " << bufferSize << " bytes\n";
            *outHandle = handle;
            std::cout << "Test handle: " << *outHandle << std::endl;

        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't create input buffer: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

int OpenCLInterface::createOutBuffer(size_t bufferSize, float *data, cl_mem *outHandle){
    try {
        cl_int result;
        cl_mem handle = clCreateBuffer(this->context, CL_MEM_WRITE_ONLY, 
                                          bufferSize, NULL, &result);

        if (result == CL_SUCCESS) {
            std::cout << "Output buffer created with size: " << bufferSize << " bytes\n";
            *outHandle = handle;
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't create output buffer: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

void OpenCLInterface::setGlobalWorkSize(size_t size){
    this->globalWorkSize = size;
}

void OpenCLInterface::setSource(const char* source, const char* name){
    this->programSource = source;
    this->programName = name;
}

int OpenCLInterface::createProgram(){
    try {
        cl_int result;
        this->program = clCreateProgramWithSource(this->context, 1, &(this->programSource), NULL, &result);
        if (result == CL_SUCCESS) {
            std::cout << "Program created\n";
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't create program: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

int OpenCLInterface::buildProgram(){
    try {
        cl_int result = clBuildProgram(this->program, 1, &(this->device), NULL, NULL, NULL);
        if (result == CL_SUCCESS) {
            std::cout << "Program built\n";
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't build program: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

int OpenCLInterface::createKernel(){
    try {
        cl_int result;
        this->kernel = clCreateKernel(program, this->programName, &result);
        if (result == CL_SUCCESS) {
            std::cout << "Kernel created\n";
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't create kernel: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

int OpenCLInterface::setAllKernelArgs(){
    for (int i = 0 ; i < this->inBuffers.size() ; i++){
        OpenCLBuffer *buffer = &this->inBuffers.at(i);
        if (setKernelArg(buffer->index, buffer->handle) != 0){
            return -1;
        };
    }

    for (int i = 0 ; i < this->outBuffers.size() ; i++){
        OpenCLBuffer *buffer = &this->outBuffers.at(i);
        if (setKernelArg(buffer->index, buffer->handle) != 0){
            return -1;
        };
    }
    return 0;
}

int OpenCLInterface::setKernelArg(const int index, cl_mem handle){
    try {
        cl_int result;
        std::cout << "Set kernel data: " << index << " " << handle << "\n";

        result = clSetKernelArg(this->kernel,
                                index,
                                sizeof(cl_mem),
                                &handle);
        if (result == CL_SUCCESS){
            std::cout << "Kernel input arg set\n";
        } else {
            std::string errorExplanation = this->getCodeExplanation(result);
            throw std::runtime_error("Couldn't set kernel input arg: " + errorExplanation);
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        this->errorEncountered = true;
        return -1;
    }
    return 0;
}

float* OpenCLInterface::getBufferDataPtr(const int index, bool isInput){
    if (isInput){
        return this->inBuffers.at(index).data;
    } else {
        return this->outBuffers.at(index).data;
    }
}

void OpenCLInterface::updateBuffer(const int index) {
    OpenCLBuffer *buffer = &this->inBuffers.at(index);
    cl_int result = clEnqueueWriteBuffer(
        this->queue,
        buffer->handle,  // Existing valid cl_mem handle
        CL_TRUE,                 // Blocking write
        0,                       // Offset
        buffer->sizeBytes, // Size in bytes
        buffer->data,                 // Host pointer with NEW data
        0, NULL, NULL
    );
    std::cout << "Wrote to buffer with result: " << getCodeExplanation(result) << std::endl;
}

void OpenCLInterface::executeAndRead(const int index){
    this->execute();
    this->readResult(index);
}

void OpenCLInterface::execute(){
    if (this->isInitialized){
        clEnqueueNDRangeKernel(this->queue, this->kernel, 1, NULL, &(this->globalWorkSize), NULL, 0, NULL, NULL);
        clFinish(queue);
    } else {
        std::cout << "Interface not initialized!\n";
    }
}

void OpenCLInterface::readResult(const int index){
    try {
        OpenCLBuffer *buffer = &this->outBuffers.at(index);
        if (buffer->isInput){
            throw std::runtime_error("Trying to read from input buffer!");
        }
        if (this->isInitialized){
            size_t bufferSize = buffer->numElements * sizeof(float);
            std::cout << "Buffer handle is: " << buffer->handle << "\n";
            cl_int result = clEnqueueReadBuffer(this->queue, buffer->handle, CL_TRUE, 0,
                                bufferSize, buffer->data, 0, NULL, NULL);
            std::cout << "Read from buffer with result: " << getCodeExplanation(result) << std::endl;
        } else {
            throw std::runtime_error("Trying to read buffer, but interface is not initialized!\n");
        }
    }
    catch (const std::exception& e){
        std::cerr << "Error: Couldn't read output buffer: " << e.what() << std::endl;
        this->errorEncountered = true;
    }
}

void OpenCLInterface::cleanup(){
    clReleaseKernel(this->kernel);
    clReleaseProgram(this->program);

    for (int i = 0 ; i < this->inBuffers.size() ; i++){
        cl_mem handle = this->inBuffers[i].handle;
        clReleaseMemObject(handle);
    }
    for (int i = 0 ; i < this->outBuffers.size() ; i++){
        cl_mem handle = this->outBuffers[i].handle;
        clReleaseMemObject(handle);
    }

    clReleaseCommandQueue(this->queue);
    clReleaseContext(this->context);
}
