#ifndef EI_CLASSIFIER_MODEL_TYPES_H_
#define EI_CLASSIFIER_MODEL_TYPES_H_

#include <stdint.h>
#include "edge-impulse-sdk/tensorflow/lite/c/common.h"
#include "edge-impulse-sdk/classifier/ei_model_types.h"

typedef struct {
    uint16_t implementation_version;
    int axes;
    float scale_axes;
    bool average;
    bool minimum;
    bool maximum;
    bool rms;
    bool stdev;
    bool skewness;
    bool kurtosis;
} ei_dsp_config_flatten_t;

typedef struct {
    uint16_t implementation_version;
    int axes;
    const char * channels;
} ei_dsp_config_image_t;

typedef struct {
    uint16_t implementation_version;
    int axes;
    int num_cepstral;
    float frame_length;
    float frame_stride;
    int num_filters;
    int fft_length;
    int win_size;
    int low_frequency;
    int high_frequency;
    float pre_cof;
    int pre_shift;
} ei_dsp_config_mfcc_t;

typedef struct {
    uint16_t implementation_version;
    int axes;
    float frame_length;
    float frame_stride;
    int num_filters;
    int fft_length;
    int low_frequency;
    int high_frequency;
    int win_size;
} ei_dsp_config_mfe_t;

typedef struct {
    uint16_t implementation_version;
    int axes;
    float scale_axes;
} ei_dsp_config_raw_t;

typedef struct {
    uint16_t implementation_version;
    int axes;
    float scale_axes;
    const char * filter_type;
    float filter_cutoff;
    int filter_order;
    int fft_length;
    int spectral_peaks_count;
    float spectral_peaks_threshold;
    const char * spectral_power_edges;
} ei_dsp_config_spectral_analysis_t;

typedef struct {
    uint16_t implementation_version;
    int axes;
    float frame_length;
    float frame_stride;
    int fft_length;
    bool show_axes;
} ei_dsp_config_spectrogram_t;

typedef struct {
    uint16_t implementation_version;
    int axes;
    float frame_length;
    float frame_stride;
    int num_filters;
    int fft_length;
    int low_frequency;
    int high_frequency;
    float pre_cof;
    bool invert_features;
} ei_dsp_config_audio_syntiant_t;

#endif // EI_CLASSIFIER_MODEL_TYPES_H_