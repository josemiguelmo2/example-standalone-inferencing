#include <stdio.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "bitmap_helper.h"

// This expects features.txt to be the result *after DSP*

size_t allocations_outside_arena = 0;

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

typedef struct cube {
    size_t col;
    size_t row;
    float confidence;
} cube_t;

float framebuffer_f32[96*96];
uint8_t framebuffer[96*96];
const size_t jpeg_buffer_size = 4096;
uint8_t jpeg_buffer[jpeg_buffer_size];
char base64_buffer[8192];

bool debug = false;


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

    if (raw_features.size() != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        printf("The size of your 'features' array is not correct. Expected %d items, but had %lu\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, raw_features.size());
        return 1;
    }

    ei_impulse_result_t result;

    signal_t signal;
    numpy::signal_from_buffer(&raw_features[0], raw_features.size(), &signal);

    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tImage resolution: %dx%d\n", EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);

    TfLiteStatus status = trained_model_init(ei_aligned_calloc);
    if (status != kTfLiteOk) {
        ei_printf("Failed to allocate TFLite arena (error code %d)\n", status);
        return 1;
    }

    TfLiteTensor *input_tensor = trained_model_input(0);
    TfLiteTensor *output_tensor = trained_model_output(0);
    if (!input_tensor || !output_tensor) {
        ei_printf("Failed to get input/output tensor\n");
        return 1;
    }

    if (input_tensor->dims->size != 4) {
        ei_printf("Invalid input_tensor dimensions, expected 4 but got %d\n", (int)input_tensor->dims->size);
        return 1;
    }

    if (input_tensor->dims->data[3] != 1) {
        ei_printf("Invalid input_tensor dimensions, expected 1 channel but got %d\n", (int)input_tensor->dims->data[3]);
        return 1;
    }

    int input_img_width = input_tensor->dims->data[1];
    int input_img_height = input_tensor->dims->data[2];

    if (input_img_width * input_img_height != EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT) {
        ei_printf("Invalid number of features, expected %d but received %d\n",
            input_img_width * input_img_height, EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT);
        return 1;
    }

    ei_printf("Input dims size %d, bytes %d\n", (int)input_tensor->dims->size, (int)input_tensor->bytes);
    for (size_t ix = 0; ix < input_tensor->dims->size; ix++) {
        ei_printf("    dim %d: %d\n", (int)ix, (int)input_tensor->dims->data[ix]);
    }
    ei_printf("Output dims size %d, bytes %d\n", (int)output_tensor->dims->size, (int)output_tensor->bytes);
    for (size_t ix = 0; ix < output_tensor->dims->size; ix++) {
        ei_printf("    dim %d: %d\n", (int)ix, (int)output_tensor->dims->data[ix]);
    }

    // one byte per value
    bool is_quantized = input_tensor->bytes == input_img_width * input_img_height;

    ei_printf("Is quantized? %d\n", is_quantized);

    uint64_t dsp_start = ei_read_timer_ms();

    bool int8_input = input_tensor->type == TfLiteType::kTfLiteInt8;
    for (size_t ix = 0; ix < raw_features.size(); ix++) {
        // Quantize the input if it is int8
        if (int8_input) {
            input_tensor->data.int8[ix] = static_cast<int8_t>(round(raw_features[ix] / input_tensor->params.scale) + input_tensor->params.zero_point);
            // printf("float %ld : %d\r\n", ix, input->data.int8[ix]);
        }
        else {
            input_tensor->data.f[ix] = raw_features[ix];
        }
    }

    uint64_t dsp_end = ei_read_timer_ms();

    uint64_t nn_start = ei_read_timer_ms();

    status = trained_model_invoke();
    if (status != kTfLiteOk) {
        ei_printf("Failed to invoke model (error code %d)\n", status);
        return 1;
    }

    uint64_t nn_end = ei_read_timer_ms();

    uint64_t post_start = ei_read_timer_ms();

    std::vector<cube_t> jan_cubes;
    std::vector<cube_t> sami_cubes;

    for (size_t row = 0; row < output_tensor->dims->data[1]; row++) {
        // ei_printf("    [ ");
        for (size_t col = 0; col < output_tensor->dims->data[2]; col++) {
            size_t loc = ((row * output_tensor->dims->data[2]) + col) * output_tensor->dims->data[3];

            float v1f, v2f, v3f;

            if (is_quantized) {
                int8_t v1 = output_tensor->data.int8[loc+0];
                int8_t v2 = output_tensor->data.int8[loc+1];
                int8_t v3 = output_tensor->data.int8[loc+2];

                float zero_point = output_tensor->params.zero_point;
                float scale = output_tensor->params.scale;

                v1f = static_cast<float>(v1 - zero_point) * scale;
                v2f = static_cast<float>(v2 - zero_point) * scale;
                v3f = static_cast<float>(v3 - zero_point) * scale;

                if (v2f < 0.1f) {
                    ei_printf("%.2f ", v2f);
                }
                else {
                    ei_printf("\033[0;33m%.2f\033[0m ", v2f);
                }
            }
            else {
                v1f = output_tensor->data.f[loc+0];
                v2f = output_tensor->data.f[loc+1];
                v3f = output_tensor->data.f[loc+2];

                if (v2f < 0.1f) {
                    ei_printf("%.2f ", v2f);
                }
                else {
                    ei_printf("\033[0;33m%.2f\033[0m ", v2f);
                }
            }

            if (v2f >= 0.5f) {
                cube_t cube = { 0 };
                cube.row = row;
                cube.col = col;
                cube.confidence = v2f;
                jan_cubes.push_back(cube);
            }
            else if (v3f >= 0.5f) {
                cube_t cube = { 0 };
                cube.row = row;
                cube.col = col;
                cube.confidence = v3f;
                sami_cubes.push_back(cube);
            }

            float v[3] = { v1f, v2f, v3f };
            // ei_printf("%f ", v[1]);

            if (v[1] > 0.3f) { // cube
                // ei_printf("1");
            }
            else {
                // ei_printf("0");
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
    // ei_printf("]\n")

    uint64_t post_end = ei_read_timer_ms();

    // ei_printf("Jan cubes:\n");
    // for (auto cube : jan_cubes) {
    //     printf("    At x=%lu, y=%lu = %.5f\n", cube.col * 8, cube.row * 8, cube.confidence);
    // }
    // ei_printf("Sami cubes:\n");
    // for (auto cube : sami_cubes) {
    //     printf("    At x=%lu, y=%lu = %.5f\n", cube.col * 8, cube.row * 8, cube.confidence);
    // }

    // make bitmap for debugging
    for (size_t ix = 0; ix < raw_features.size(); ix++) {
        uint8_t pixel = (uint8_t)(raw_features[ix] * 255.0f);
        int32_t rgb_v = (pixel << 16) + (pixel << 8) + pixel;
        raw_features[ix] = (float)rgb_v;
    }

    for (auto cube : jan_cubes) {
        for (size_t offset_r = 0; offset_r < 8; offset_r++) {
            for (size_t offset_c = 0; offset_c < 8; offset_c++) {
                // todo: handle overflow to next row here
                raw_features[(((cube.row * 8) + offset_r) * input_img_width) + ((cube.col * 8) + offset_c)] = (float)0xff0000;
            }
        }
        // raw_features[(cube.row * 8 * input_img_width) + (cube.col * 8)] = (float)0xff0000;
    }
    for (auto cube : sami_cubes) {
        for (size_t offset_r = 0; offset_r < 8; offset_r++) {
            for (size_t offset_c = 0; offset_c < 8; offset_c++) {
                // todo: handle overflow to next row here
                raw_features[(((cube.row * 8) + offset_r) * input_img_width) + ((cube.col * 8) + offset_c)] = (float)0x00ff00;
            }
        }
        // raw_features[(cube.row * 8 * input_img_width) + (cube.col * 8)] = (float)0xff0000;
    }
    create_bitmap_file("debug.bmp", raw_features.data(), input_img_width, input_img_height);

    status = trained_model_reset(ei_aligned_free);
    if (status != kTfLiteOk) {
        ei_printf("Failed to free model (error code %d)\n", status);
        return 1;
    }

//     EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
//     printf("run_classifier returned: %d\n", res);

//     printf("Begin output\n");

// #if EI_CLASSIFIER_OBJECT_DETECTION == 1
//     for (size_t ix = 0; ix < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; ix++) {
//         auto bb = result.bounding_boxes[ix];
//         if (bb.value == 0) {
//             continue;
//         }

//         printf("%s (%f) [ x: %u, y: %u, width: %u, height: %u ]\n", bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
//     }
// #else
//     // print the predictions
//     printf("[");
//     for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
//         printf("%.5f", result.classification[ix].value);
// #if EI_CLASSIFIER_HAS_ANOMALY == 1
//         printf(", ");
// #else
//         if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
//             printf(", ");
//         }
// #endif
//     }
// #if EI_CLASSIFIER_HAS_ANOMALY == 1
//     printf("%.3f", result.anomaly);
// #endif
//     printf("]\n");
// #endif

//     printf("End output\n");

//     ei_printf("Allocations outside buffer: %lu bytes\n", allocations_outside_arena);
}
