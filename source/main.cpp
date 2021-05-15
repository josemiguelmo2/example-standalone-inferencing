#include <stdio.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "model-parameters/impulse1_model_metadata.h"

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

void print_impulse_result(ei_impulse_t *impulse, ei_impulse_result_t *result) {
    if (impulse->object_detection) {
        for (size_t ix = 0; ix < impulse->object_detection_count; ix++) {
            auto bb = result->bounding_boxes[ix];
            if (bb.value == 0) {
                continue;
            }

            printf("%s (%f) [ x: %u, y: %u, width: %u, height: %u ]\n", bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
        }
    }
    else {
        // print the predictions
        printf("[");
        for (size_t ix = 0; ix < impulse->label_count; ix++) {
            printf("%.5f", result->classification[ix].value);
            if (impulse->has_anomaly) {
                printf(", ");
            }
            else {
                if (ix != impulse->label_count - 1) {
                    printf(", ");
                }
            }
        }
        if (impulse->has_anomaly) {
            printf("%.3f", result->anomaly);
        }
        printf("]\n");
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Requires one parameter (a comma-separated list of raw features, or a file pointing at raw features)\n");
        return 1;
    }

    if (impulse_1.object_detection) {
        printf("WARN: For object detection models, consider https://github.com/edgeimpulse/example-standalone-inferencing-linux which has full hardware acceleration\n");
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

    if (raw_features.size() != impulse_1.dsp_input_frame_size) {
        printf("The size of your 'features' array is not correct. Expected %d items, but had %lu\n",
            impulse_1.dsp_input_frame_size, raw_features.size());
        return 1;
    }

    ei_impulse_result_t result_1;

    signal_t signal;
    numpy::signal_from_buffer(&raw_features[0], raw_features.size(), &signal);

    EI_IMPULSE_ERROR res = run_classifier(&impulse_1, &signal, &result_1, true);
    printf("run_classifier returned: %d\n", res);

    printf("Begin output\n");
    print_impulse_result(&impulse_1, &result_1);
    printf("End output\n");
}
