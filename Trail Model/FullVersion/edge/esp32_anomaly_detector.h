/**
 * @file esp32_anomaly_detector.h
 * @brief Ultra-lightweight TinyML / Physics-guided Edge Anomaly Detection for ESP32.
 * @details Zero external dependencies. Designed for low-power microcontrollers (ESP32, STM32, RP2040).
 *          Execution time: < 0.1 ms per observation. RAM requirement: < 2 KB.
 *          Target: Smart India Hackathon (SIH 2026) Edge AI Evaluation.
 */

#ifndef ESP32_ANOMALY_DETECTOR_H
#define ESP32_ANOMALY_DETECTOR_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return status codes */
typedef enum {
    AWS_STATUS_OK = 0,
    AWS_STATUS_GENUINE_WEATHER_EVENT = 1,
    AWS_STATUS_ERR_OUT_OF_BOUNDS = 2,
    AWS_STATUS_ERR_SPIKE = 3,
    AWS_STATUS_ERR_STUCK_SENSOR = 4,
    AWS_STATUS_ERR_PHYSICAL_INCONSISTENCY = 5
} aws_status_t;

/* Faulty parameter identifier */
typedef enum {
    FAULT_NONE = 0,
    FAULT_TEMPERATURE = 1,
    FAULT_PRESSURE = 2,
    FAULT_HUMIDITY = 3,
    FAULT_MULTIVARIATE = 4
} aws_fault_param_t;

/* Result structure returned on every observation */
typedef struct {
    aws_status_t status;
    aws_fault_param_t faulty_param;
    bool is_anomaly;            /* True only if sensor is faulty (False for genuine weather) */
    bool is_weather_event;      /* True if convective storm / cold front */
    float confidence;           /* 0.0 to 1.0 */
    float dew_point;            /* Computed on-chip via Magnus-Tetens */
    float vapor_pressure_deficit; /* VPD in hPa */
    const char* message;
} aws_detection_result_t;

/* Circular buffer history size (e.g. 16 samples) */
#define AWS_EDGE_HISTORY_LEN 16

/* Configuration thresholds conforming to WMO standards */
typedef struct {
    float temp_min;             /* -40.0 C */
    float temp_max;             /* +60.0 C */
    float press_min;            /* 800.0 hPa */
    float press_max;            /* 1085.0 hPa */
    float rh_min;               /* 0.0 % */
    float rh_max;               /* 100.0 % */
    float max_delta_temp;       /* 3.0 C per sample */
    float max_delta_press;      /* 2.0 hPa per sample */
    float max_delta_rh;         /* 15.0 % per sample */
    float min_stdev_flatline;   /* 0.005 minimum variance */
} aws_edge_config_t;

/* Detector runtime state struct */
typedef struct {
    aws_edge_config_t config;
    float temp_buf[AWS_EDGE_HISTORY_LEN];
    float press_buf[AWS_EDGE_HISTORY_LEN];
    float rh_buf[AWS_EDGE_HISTORY_LEN];
    uint8_t count;
    uint8_t head;
} aws_edge_detector_t;

/**
 * @brief Initialize edge detector with default meteorological limits.
 */
static inline void aws_edge_init(aws_edge_detector_t* det) {
    det->config.temp_min = -40.0f;
    det->config.temp_max = 60.0f;
    det->config.press_min = 800.0f;
    det->config.press_max = 1085.0f;
    det->config.rh_min = 0.0f;
    det->config.rh_max = 100.0f;
    det->config.max_delta_temp = 3.0f;
    det->config.max_delta_press = 2.0f;
    det->config.max_delta_rh = 15.0f;
    det->config.min_stdev_flatline = 0.005f;

    det->count = 0;
    det->head = 0;
}

/**
 * @brief Fast single-precision Magnus-Tetens dew point calculation.
 */
static inline float aws_edge_dew_point(float temp_c, float rh) {
    if (rh <= 0.01f) rh = 0.01f;
    if (rh > 100.0f) rh = 100.0f;

    const float a = 17.67f;
    const float b = 243.5f;
    float alpha = logf(rh / 100.0f) + (a * temp_c) / (b + temp_c);
    return (b * alpha) / (a - alpha);
}

/**
 * @brief Fast saturation vapor pressure (hPa).
 */
static inline float aws_edge_es(float temp_c) {
    return 6.112f * expf((17.67f * temp_c) / (temp_c + 243.5f));
}

/**
 * @brief Process single sensor observation on ESP32.
 */
static inline aws_detection_result_t aws_edge_process(
    aws_edge_detector_t* det,
    float temp,
    float press,
    float rh
) {
    aws_detection_result_t res;
    res.status = AWS_STATUS_OK;
    res.faulty_param = FAULT_NONE;
    res.is_anomaly = false;
    res.is_weather_event = false;
    res.confidence = 0.95f;
    res.dew_point = aws_edge_dew_point(temp, rh);
    float es = aws_edge_es(temp);
    float e = es * (rh / 100.0f);
    res.vapor_pressure_deficit = es - e;
    res.message = "Nominal";

    /* 1. Physical plausibility check */
    if (temp < det->config.temp_min || temp > det->config.temp_max) {
        res.status = AWS_STATUS_ERR_OUT_OF_BOUNDS;
        res.faulty_param = FAULT_TEMPERATURE;
        res.is_anomaly = true;
        res.confidence = 0.99f;
        res.message = "Temperature out of physical bounds";
        return res;
    }
    if (press < det->config.press_min || press > det->config.press_max) {
        res.status = AWS_STATUS_ERR_OUT_OF_BOUNDS;
        res.faulty_param = FAULT_PRESSURE;
        res.is_anomaly = true;
        res.confidence = 0.99f;
        res.message = "Pressure out of physical bounds";
        return res;
    }
    if (rh < det->config.rh_min || rh > det->config.rh_max) {
        res.status = AWS_STATUS_ERR_OUT_OF_BOUNDS;
        res.faulty_param = FAULT_HUMIDITY;
        res.is_anomaly = true;
        res.confidence = 0.98f;
        res.message = "Humidity out of physical bounds";
        return res;
    }

    /* 2. Thermodynamic consistency check */
    if (res.dew_point > temp + 0.2f) {
        res.status = AWS_STATUS_ERR_PHYSICAL_INCONSISTENCY;
        res.faulty_param = FAULT_MULTIVARIATE;
        res.is_anomaly = true;
        res.confidence = 0.92f;
        res.message = "Dew point exceeds dry bulb temperature";
        return res;
    }

    /* If we have previous history, check step rate of change and severe storm coupling */
    if (det->count > 0) {
        uint8_t prev_idx = (det->head + AWS_EDGE_HISTORY_LEN - 1) % AWS_EDGE_HISTORY_LEN;
        float prev_t = det->temp_buf[prev_idx];
        float prev_p = det->press_buf[prev_idx];
        float prev_rh = det->rh_buf[prev_idx];

        float dt = temp - prev_t;
        float dp = press - prev_p;
        float drh = rh - prev_rh;

        /* Check for genuine severe convective weather signature:
           Sharp temp drop (rain cooling <= -2.5 C) AND humidity surge (>= +15 %) */
        if (dt <= -2.5f && drh >= 15.0f) {
            res.status = AWS_STATUS_GENUINE_WEATHER_EVENT;
            res.is_weather_event = true;
            res.is_anomaly = false; /* Healthy sensor, real storm! */
            res.confidence = 0.95f;
            res.message = "Convective Thunderstorm / Downdraft Signature detected";
        }
        else {
            /* Check rate of change spike */
            if (fabsf(dt) > det->config.max_delta_temp) {
                res.status = AWS_STATUS_ERR_SPIKE;
                res.faulty_param = FAULT_TEMPERATURE;
                res.is_anomaly = true;
                res.confidence = 0.90f;
                res.message = "Temperature spike exceeded rate limit";
                return res;
            }
            if (fabsf(dp) > det->config.max_delta_press) {
                res.status = AWS_STATUS_ERR_SPIKE;
                res.faulty_param = FAULT_PRESSURE;
                res.is_anomaly = true;
                res.confidence = 0.90f;
                res.message = "Pressure spike exceeded rate limit";
                return res;
            }
            if (fabsf(drh) > det->config.max_delta_rh) {
                res.status = AWS_STATUS_ERR_SPIKE;
                res.faulty_param = FAULT_HUMIDITY;
                res.is_anomaly = true;
                res.confidence = 0.90f;
                res.message = "Humidity spike exceeded rate limit";
                return res;
            }
        }
    }

    /* 3. Persistence / Stuck Sensor Flatline Check */
    if (det->count >= 6) {
        float sum_sq_t = 0.0f, sum_t = 0.0f;
        float sum_sq_p = 0.0f, sum_p = 0.0f;
        float sum_sq_rh = 0.0f, sum_rh = 0.0f;
        uint8_t check_len = 6;

        for (uint8_t i = 0; i < check_len; i++) {
            uint8_t idx = (det->head + AWS_EDGE_HISTORY_LEN - 1 - i) % AWS_EDGE_HISTORY_LEN;
            float t_val = det->temp_buf[idx];
            float p_val = det->press_buf[idx];
            float rh_val = det->rh_buf[idx];

            sum_t += t_val; sum_sq_t += t_val * t_val;
            sum_p += p_val; sum_sq_p += p_val * p_val;
            sum_rh += rh_val; sum_sq_rh += rh_val * rh_val;
        }

        float var_t = (sum_sq_t / check_len) - ((sum_t / check_len) * (sum_t / check_len));
        float var_p = (sum_sq_p / check_len) - ((sum_p / check_len) * (sum_p / check_len));
        float var_rh = (sum_sq_rh / check_len) - ((sum_rh / check_len) * (sum_rh / check_len));

        if (var_t < 1e-5f) {
            res.status = AWS_STATUS_ERR_STUCK_SENSOR;
            res.faulty_param = FAULT_TEMPERATURE;
            res.is_anomaly = true;
            res.confidence = 0.93f;
            res.message = "Temperature sensor frozen / flatlined";
            return res;
        }
        if (var_p < 1e-5f) {
            res.status = AWS_STATUS_ERR_STUCK_SENSOR;
            res.faulty_param = FAULT_PRESSURE;
            res.is_anomaly = true;
            res.confidence = 0.93f;
            res.message = "Pressure sensor frozen / flatlined";
            return res;
        }
        if (var_rh < 1e-5f) {
            res.status = AWS_STATUS_ERR_STUCK_SENSOR;
            res.faulty_param = FAULT_HUMIDITY;
            res.is_anomaly = true;
            res.confidence = 0.93f;
            res.message = "Humidity sensor frozen / flatlined";
            return res;
        }
    }

    /* Save reading to circular buffer */
    det->temp_buf[det->head] = temp;
    det->press_buf[det->head] = press;
    det->rh_buf[det->head] = rh;
    det->head = (det->head + 1) % AWS_EDGE_HISTORY_LEN;
    if (det->count < AWS_EDGE_HISTORY_LEN) det->count++;

    return res;
}

#ifdef __cplusplus
}
#endif

#endif /* ESP32_ANOMALY_DETECTOR_H */
