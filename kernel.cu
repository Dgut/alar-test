#include "cuda_runtime.h"
#include <stdio.h>

const int LZ77windowBits = 11;
const int LZ77matchBits = 16 - LZ77windowBits;
const int LZ77windowMask = (1 << LZ77windowBits) - 1;
const int LZ77matchMask = (1 << LZ77matchBits) - 1;
const int LZ77windowSize = 1 << LZ77windowBits;
const int LZ77matchSize = 1 << LZ77matchBits;

__device__ int equalLength(const int* a, const int* b, const int limit)
{
    int result = 0;

    while (result < limit)
        if (*a++ == *b++)
            result++;
        else
            break;

    return result;
}

__global__ void LZ77match(const int* input, int* offset, int* length, const int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size)
    {
        const int limit = min(size - i - 1, LZ77matchSize);
        const int maxOffset = min(i, LZ77windowSize);

        offset[i] = 0;
        length[i] = 0;

        for (int o = 1; o <= maxOffset; o++)
        {
            const int l = equalLength(input + i, input + i - o, limit);

            if (length[i] < l)
            {
                length[i] = l;
                offset[i] = o;
            }
        }
    }
}

__global__ void LZ77encode(const int* input, int* offset, int* length, const int inputSize, unsigned char* output, int* outputSize)
{
    *outputSize = 0;

    for (int i = 0; i < inputSize; i++)
    {
        if (length[i] > 2)
        {
            const unsigned short block = ((offset[i] - 1) & LZ77windowMask) | (((length[i] - 1) & LZ77matchMask) << LZ77windowBits);

            i += length[i];

            output[(*outputSize)++] = input[i] & 0x7f | 0x80;
            output[(*outputSize)++] = block;
            output[(*outputSize)++] = block >> 8;
        }
        else
        {
            output[(*outputSize)++] = input[i] & 0x7f;
        }
    }
}

__global__ void LZ77decode(const unsigned char* input, int inputSize, int* output, int* outputSize)
{
    int* start = output;
    const unsigned char* end = input + inputSize;

    while (input < end)
    {
        if (*input & 0x80)
        {
            const unsigned short block = input[1] | (input[2] << 8);

            int* base = output - ((block & LZ77windowMask) + 1);
            int length = ((block >> LZ77windowBits) & LZ77windowMask) + 1;

            while(length-- > 0)
            {
                *output = *base;
                output++;
                base++;
            }

            *output = *input & 0x7f;
            output++;

            input += 3;
        }
        else
        {
            *output = *input;
            output++;

            input++;
        }
    }

    *outputSize = output - start;
}

const int maxInputSize = 1000000;
int intData[maxInputSize];
unsigned char byteData[maxInputSize];
int intSize = 0;
int byteSize = 0;

void readDecodedData(const char* file, int* data, int* size)
{
    FILE* stream = fopen(file, "rt");

    *size = 0;
    while (fscanf(stream, "%i", data + *size) > 0)
        (*size)++;

    fclose(stream);
}

void writeDecodedData(const char* file, const int* data, const int size)
{
    FILE* stream = fopen(file, "wt");

    for (int i = 0; i < size; i++)
        fprintf(stream, "%i\n", data[i]);

    fclose(stream);
}

void readEncodedData(const char* file, unsigned char* data, int* size)
{
    FILE* stream = fopen(file, "rb");

    *size = 0;
    while (!feof(stream))
        *size += fread(data + *size, 1, 1024, stream);

    fclose(stream);
}

void writeEncodedData(const char* file, const unsigned char* data, const int size)
{
    FILE* stream = fopen(file, "wb");

    fwrite(data, 1, size, stream);

    fclose(stream);
}

int main(int argc, const char** argv)
{
    if (argc < 4)
        return -1;

    if (!strcmp(argv[1], "-e"))
    {
        readDecodedData(argv[2], intData, &intSize);

        int* dev_int;
        int* dev_length;
        int* dev_offset;
        unsigned char* dev_byte;
        int* dev_size;

        cudaMalloc(&dev_int, intSize * sizeof(int));
        cudaMalloc(&dev_length, intSize * sizeof(int));
        cudaMalloc(&dev_offset, intSize * sizeof(int));
        cudaMalloc(&dev_byte, intSize);
        cudaMalloc(&dev_size, sizeof(int));

        cudaMemcpy(dev_int, intData, intSize * sizeof(int), cudaMemcpyHostToDevice);

        LZ77match<<<(intSize + 511) / 512, 512>>>(dev_int, dev_offset, dev_length, intSize);
        LZ77encode<<<1, 1>>>(dev_int, dev_offset, dev_length, intSize, dev_byte, dev_size);

        cudaDeviceSynchronize();

        cudaMemcpy(&byteSize, dev_size, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(byteData, dev_byte, byteSize, cudaMemcpyDeviceToHost);

        cudaFree(dev_int);
        cudaFree(dev_length);
        cudaFree(dev_offset);
        cudaFree(dev_byte);
        cudaFree(dev_size);

        cudaDeviceReset();

        writeEncodedData(argv[3], byteData, byteSize);
    }
    else if (!strcmp(argv[1], "-d"))
    {
        readEncodedData(argv[2], byteData, &byteSize);

        unsigned char* dev_byte;
        int* dev_int;
        int* dev_size;

        cudaMalloc(&dev_byte, byteSize);
        cudaMalloc(&dev_int, maxInputSize * sizeof(int));
        cudaMalloc(&dev_size, sizeof(int));

        cudaMemcpy(dev_byte, byteData, byteSize, cudaMemcpyHostToDevice);

        LZ77decode<<<1, 1>>>(dev_byte, byteSize, dev_int, dev_size);

        cudaDeviceSynchronize();

        cudaMemcpy(&intSize, dev_size, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(intData, dev_int, intSize * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(dev_byte);
        cudaFree(dev_int);
        cudaFree(dev_size);

        cudaDeviceReset();

        writeDecodedData(argv[3], intData, intSize);
    }
    else
        return -1;

    return 0;
}
