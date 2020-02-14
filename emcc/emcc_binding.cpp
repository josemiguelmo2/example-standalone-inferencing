#include <emscripten/bind.h>
#include "ei_run_classifier.h"

using namespace emscripten;

typedef struct {
    std::string label;
    float value;
} emcc_classification_result_category_t;

typedef struct {
    int result;
    std::vector<emcc_classification_result_category_t> classification;
    float anomaly;
} emcc_classification_result_t;

emcc_classification_result_t emcc_run_classifier(size_t input_buffer_raw, size_t input_buffer_size, bool debug) {
    float *input_buffer = (float*)input_buffer_raw;

    ei_impulse_result_t impulse_result;

    signal_t signal;
    numpy::signal_from_buffer(input_buffer, input_buffer_size, &signal);

    EI_IMPULSE_ERROR res = run_classifier(&signal, &impulse_result, debug);

    emcc_classification_result_t ret;
    ret.result = (int)res;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        emcc_classification_result_category_t r;
        r.label = std::string(impulse_result.classification[ix].label);
        r.value = impulse_result.classification[ix].value;

        ret.classification.push_back(r);
    }
    ret.anomaly = impulse_result.anomaly;

    return ret;
}

EMSCRIPTEN_BINDINGS(my_module) {
    class_<emcc_classification_result_category_t>("emcc_classification_result_category_t")
        .constructor<>()
        .property("label", &emcc_classification_result_category_t::label)
        .property("value", &emcc_classification_result_category_t::value);

    class_<emcc_classification_result_t>("emcc_classification_result_t")
        .constructor<>()
        .property("result", &emcc_classification_result_t::result)
        .property("classification", &emcc_classification_result_t::classification)
        .property("anomaly", &emcc_classification_result_t::anomaly)
        ;

    function("run_classifier", &emcc_run_classifier, allow_raw_pointers());

    register_vector<emcc_classification_result_category_t>("vector<emcc_classification_result_category_t>");
}
