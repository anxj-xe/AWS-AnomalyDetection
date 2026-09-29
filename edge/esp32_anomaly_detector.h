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
    float max_delta_temp;       /* Rate limit per interval (e.g. 6.0 C / 15m, 2.5 C / 1m) */
    float max_delta_press;      /* 4.0 hPa / 15m, 1.8 hPa / 1m */
    float max_delta_rh;         /* 25.0 % / 15m, 15.0 % / 1m */
    float min_stdev_flatline;   /* 0.005 minimum variance */
    uint8_t interval_minutes;   /* 15 (routine synoptic) or 1 (rapid storm) */
} aws_edge_config_t;

/* Detector runtime state struct */
typedef struct {
    aws_edge_config_t config;
    float temp_buf[AWS_EDGE_HISTORY_LEN];
    float press_buf[AWS_EDGE_HISTORY_LEN];
    float rh_buf[AWS_EDGE_HISTORY_LEN];
    uint8_t count;
    uint8_t head;
    bool in_storm_mode;             /* True when operating in rapid 1-minute storm mode */
    uint8_t stable_cooldown_count;  /* Consecutive quiet 1-minute samples required to confirm storm is over */
} aws_edge_detector_t;

/**
 * @brief Configure sampling interval and adaptive WMO rate thresholds.
 * @param interval_minutes 15 for routine synoptic mode, 1 for rapid storm mode.
 */
static inline void aws_edge_set_interval(aws_edge_detector_t* det, uint8_t interval_minutes) {
    det->config.interval_minutes = interval_minutes;
    if (interval_minutes <= 2) {
        /* High-frequency storm mode (1-minute) */
        det->config.max_delta_temp = 2.5f;
        det->config.max_delta_press = 1.8f;
        det->config.max_delta_rh = 15.0f;
    } else {
        /* Routine synoptic mode (15-minute) */
        det->config.max_delta_temp = 6.0f;
        det->config.max_delta_press = 4.0f;
        det->config.max_delta_rh = 25.0f;
    }
}

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
    det->config.min_stdev_flatline = 0.005f;
    aws_edge_set_interval(det, 15); /* Default to 15-minute synoptic routine */

    det->count = 0;
    det->head = 0;
    det->in_storm_mode = false;
    det->stable_cooldown_count = 0;
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
            det->in_storm_mode = true;
            det->stable_cooldown_count = 0; /* Reset cooldown timer on active storm pulse */
            aws_edge_set_interval(det, 1);  /* Switch to 1-minute high-frequency storm mode */
        }
        else {
            /* If operating in rapid storm mode, test for atmospheric stabilization (storm over) */
            if (det->in_storm_mode) {
                /* Check if atmosphere has ceased rapid fluctuation (calm/settled) */
                if (fabsf(dt) <= 0.4f && fabsf(dp) <= 0.3f && fabsf(drh) <= 2.5f) {
                    det->stable_cooldown_count++;
                    if (det->stable_cooldown_count >= 15) {
                        /* 15 consecutive minutes of calm confirmed storm has fully dissipated */
                        det->in_storm_mode = false;
                        det->stable_cooldown_count = 0;
                        aws_edge_set_interval(det, 15); /* Revert back to standard 15-minute synoptic mode */
                        res.message = "Storm Over: Atmosphere stabilized for 15m; reverted to 15-min synoptic mode";
                    }
                } else {
                    /* Secondary squall or turbulence detected - reset stabilization counter */
                    det->stable_cooldown_count = 0;
                }
            }

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
    if (det->count >= 3) {
        float min_t = temp, max_t = temp;
        float min_p = press, max_p = press;
        float min_rh = rh, max_rh = rh;
        float sum_sq_t = temp * temp, sum_t = temp;
        float sum_sq_p = press * press, sum_p = press;
        float sum_sq_rh = rh * rh, sum_rh = rh;
        uint8_t check_len = (det->count >= 4) ? 5 : 4;

        for (uint8_t i = 0; i < check_len - 1; i++) {
            uint8_t idx = (det->head + AWS_EDGE_HISTORY_LEN - 1 - i) % AWS_EDGE_HISTORY_LEN;
            float t_val = det->temp_buf[idx];
            float p_val = det->press_buf[idx];
            float rh_val = det->rh_buf[idx];

            if (t_val < min_t) min_t = t_val; if (t_val > max_t) max_t = t_val;
            if (p_val < min_p) min_p = p_val; if (p_val > max_p) max_p = p_val;
            if (rh_val < min_rh) min_rh = rh_val; if (rh_val > max_rh) max_rh = rh_val;

            sum_t += t_val; sum_sq_t += t_val * t_val;
            sum_p += p_val; sum_sq_p += p_val * p_val;
            sum_rh += rh_val; sum_sq_rh += rh_val * rh_val;
        }

        float var_t = (sum_sq_t / check_len) - ((sum_t / check_len) * (sum_t / check_len));
        float var_p = (sum_sq_p / check_len) - ((sum_p / check_len) * (sum_p / check_len));
        float var_rh = (sum_sq_rh / check_len) - ((sum_rh / check_len) * (sum_rh / check_len));

        if ((max_t - min_t) < 1e-4f || (check_len >= 5 && var_t < 2.5e-5f)) {
            res.status = AWS_STATUS_ERR_STUCK_SENSOR;
            res.faulty_param = FAULT_TEMPERATURE;
            res.is_anomaly = true;
            res.confidence = 0.95f;
            res.message = "Temperature sensor frozen / flatlined";
            return res;
        }
        if ((max_p - min_p) < 1e-4f || (check_len >= 5 && var_p < 2.5e-5f)) {
            res.status = AWS_STATUS_ERR_STUCK_SENSOR;
            res.faulty_param = FAULT_PRESSURE;
            res.is_anomaly = true;
            res.confidence = 0.95f;
            res.message = "Pressure sensor frozen / flatlined";
            return res;
        }
        if ((max_rh - min_rh) < 1e-4f || (check_len >= 5 && var_rh < 2.25e-4f)) {
            res.status = AWS_STATUS_ERR_STUCK_SENSOR;
            res.faulty_param = FAULT_HUMIDITY;
            res.is_anomaly = true;
            res.confidence = 0.95f;
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
