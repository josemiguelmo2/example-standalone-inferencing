#include <stdio.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
#include "edge-impulse-sdk/classifier/ei_aligned_malloc.h"
#include "tflite-model/20210903_123610.float32.cpp.h"
#include "bitmap_helper.h"

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
#warning "For object detection models, consider https://github.com/edgeimpulse/example-standalone-inferencing-linux which has full hardware acceleration"
#endif

static void softmax(float *input, size_t input_len) {
  assert(input);
  // assert(input_len >= 0);  Not needed

  float m = -INFINITY;
  for (size_t i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (size_t i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  float offset = m + logf(sum);
  for (size_t i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}

typedef struct cube {
    size_t col;
    size_t row;
} cube_t;

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

std::string read_file(const char *filename) {
    FILE *f = (FILE*)fopen(filename, "r");
    if (!f) {
        printf("Cannot open file %s\n", filename);
        return "";
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    std::string ss;
    ss.resize(size);
    rewind(f);
    fread(&ss[0], 1, size, f);
    fclose(f);
    return ss;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Requires one parameter (a comma-separated list of raw features, or a file pointing at raw features)\n");
        return 1;
    }

    std::string input = argv[1];
    if (!strchr(argv[1], ' ') && strchr(argv[1], '.')) { // looks like a filename
        input = read_file(argv[1]);
    }

    std::istringstream ss(input);
    std::string token;

    std::vector<float> raw_features;

    while (std::getline(ss, token, ',')) {
        raw_features.push_back(std::stof(trim(token)));
    }

    ei_printf("Have features: %d\n", (int)raw_features.size());

    TfLiteStatus status = model_init(ei_aligned_calloc);
    if (status != kTfLiteOk) {
        ei_printf("Failed to allocate TFLite arena (error code %d)\n", status);
        return EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED;
    }

    TfLiteTensor *input_tensor = model_input(0);
    TfLiteTensor *output_tensor = model_output(0);
    if (!input_tensor || !output_tensor) {
        ei_printf("Failed to get input/output tensor\n");
        return 1;
    }

    for (size_t ix = 0; ix < raw_features.size(); ix++) {
        // input_tensor->data.int8[ix] = static_cast<int8_t>(round(raw_features.at(ix) / input_tensor->params.scale) + input_tensor->params.zero_point);
        input_tensor->data.f[ix] = raw_features.at(ix);
    }

    status = model_invoke();
    if (status != kTfLiteOk) {
        ei_printf("Failed to invoke model (error code %d)\n", status);
        return 1;
    }

    ei_printf("Output dims size %d, bytes %d\n", (int)output_tensor->dims->size, (int)output_tensor->bytes);
    for (size_t ix = 0; ix < output_tensor->dims->size; ix++) {
        ei_printf("dim %d: %d\n", (int)ix, (int)output_tensor->dims->data[ix]);
    }

    // ei_printf("[\n");

    std::vector<cube_t> cubes;

    for (size_t row = 0; row < output_tensor->dims->data[1]; row++) {
        // ei_printf("    [ ");
        for (size_t col = 0; col < output_tensor->dims->data[2]; col++) {
            size_t loc = ((row * output_tensor->dims->data[2]) + col) * 2;

            // int8_t v1 = output_tensor->data.int8[loc+0];
            // int8_t v2 = output_tensor->data.int8[loc+1];

            // float zero_point = output_tensor->params.zero_point;
            // float scale = output_tensor->params.scale;

            // float v1f = static_cast<float>(v1 - zero_point) * scale;
            // float v2f = static_cast<float>(v2 - zero_point) * scale;

            float v1f = output_tensor->data.f[loc+0];
            float v2f = output_tensor->data.f[loc+1];

            float v[2] = { v1f, v2f };
            // softmax(v, 2);

            if (v[1] > 0.3f) { // cube
                cube_t cube = { 0 };
                cube.row = row;
                cube.col = col;
                bool found_overlapping_cube = false;
                for (auto other_cube : cubes) {
                    if (abs((int)(cube.row - other_cube.row)) <= 1 && abs((int)(cube.col - other_cube.col)) <= 1) {
                        // overlapping
                        found_overlapping_cube = true;
                    }
                }
                if (!found_overlapping_cube) {
                    cubes.push_back(cube);
                }

                ei_printf("1");
            }
            else {
                ei_printf("0");
            }

            // ei_printf("%.2f", v[1]);

            // ei_printf("[ %.2f, %.2f ]", v[0], v[1]);
            // ei_printf("[ %f, %f ]", v1f, v2f);
            if (col != output_tensor->dims->data[2] - 1) {
                // ei_printf(", ");
            }
        }
        // ei_printf("]");
        if (row != output_tensor->dims->data[1] - 1) {
            // ei_printf(", ");
        }
        ei_printf("\n");
    }
    // ei_printf("]\n");

    status = model_reset(ei_aligned_free);
    if (status != kTfLiteOk) {
        ei_printf("Failed to free model (error code %d)\n", status);
        return 1;
    }

    ei_printf("Found %lu cubes\n", cubes.size());
    for (auto cube : cubes) {
        ei_printf("    At x=%lu, y=%lu\n", cube.col * 8, cube.row * 8);
    }

    // make bitmap for debugging
    for (size_t ix = 0; ix < raw_features.size(); ix++) {
        uint8_t pixel = (uint8_t)(raw_features[ix] * 255.0f);
        int32_t rgb_v = (pixel << 16) + (pixel << 8) + pixel;
        raw_features[ix] = (float)rgb_v;
    }

    for (auto cube : cubes) {
        for (size_t offset_r = 0; offset_r < 8; offset_r++) {
            for (size_t offset_c = 0; offset_c < 8; offset_c++) {
                // todo: handle overflow to next row here
                raw_features[(((cube.row * 8) + offset_r) * 320) + ((cube.col * 8) + offset_c)] = (float)0xff0000;
            }
        }
        // raw_features[(cube.row * 8 * 320) + (cube.col * 8)] = (float)0xff0000;
    }
    create_bitmap_file("debug.bmp", raw_features.data(), 320, 320);

    ei_printf("Running model OK\n");
}
