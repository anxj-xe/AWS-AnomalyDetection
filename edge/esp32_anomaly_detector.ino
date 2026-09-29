/**
 * @file esp32_anomaly_detector.ino
 * @brief ESP32 firmware sketch for real-time edge anomaly screening.
 * 
 * Hardware: ESP32 / ESP32-S3 / ESP32-C3
 * Sensors: BME280 / SHT3x or analog RTD bridge
 */

#include <Arduino.h>
#include "esp32_anomaly_detector.h"

// Instantiate edge detector state
static aws_edge_detector_t edge_detector;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n-------------------------------------------------------");
  Serial.println("  AWS Edge Telemetry Monitor (ESP32)");
  Serial.println("  WMO-No. 8 Standard Quality Control Initialized");
  Serial.println("-------------------------------------------------------\n");

  aws_edge_init(&edge_detector);
}

// Function to simulate incoming sensor read (e.g. from I2C BME280)
void read_simulated_aws_sensors(float *out_t, float *out_p, float *out_rh, uint32_t step) {
  // Base diurnal cycle (96 intervals of 15 minutes per 24-hour day)
  float hour = (step % 96) / 4.0; // 0.0 to 24.0 hours
  float angle = 2.0 * 3.14159 * (hour - 8.5) / 24.0;
  
  *out_t = 28.0 + 6.0 * sin(angle) + ((rand() % 20 - 10) / 100.0);
  *out_rh = 65.0 - 20.0 * sin(angle) + ((rand() % 40 - 20) / 100.0);
  *out_p = 1012.0 + 1.5 * cos(4.0 * 3.14159 * hour / 24.0);

  // Injected Faults for demonstration:
  if (step == 15) {
    // Inject sudden temperature spike (+12.5 C in 15 minutes)
    *out_t += 12.5;
  } else if (step >= 25 && step <= 28) {
    // Inject frozen humidity sensor
    *out_rh = 72.4;
  } else if (step == 45) {
    // Inject Genuine Severe Thunderstorm Event (Cold downdraft + humidity surge)
    *out_t -= 5.5;
    *out_rh += 28.0;
    *out_p -= 2.2;
  }
}

void loop() {
  static uint32_t step_counter = 0;

  float current_temp, current_press, current_rh;
  read_simulated_aws_sensors(&current_temp, &current_press, &current_rh, step_counter);

  // Measure execution latency on ESP32 (<0.08 ms)
  uint32_t t_start = micros();
  aws_detection_result_t result = aws_edge_process(&edge_detector, current_temp, current_press, current_rh);
  uint32_t elapsed_us = micros() - t_start;

  // Adaptive Frequency Trigger & Recovery:
  // aws_edge_process autonomously manages the storm lifecycle:
  // 1. When convective storm signature is detected, switches to 1-minute rapid mode.
  // 2. Monitors atmospheric calm across a 15-minute hysteresis window.
  // 3. When 15 consecutive minutes of calm are confirmed, automatically reverts to 15-minute synoptic mode!
  if (result.is_weather_event) {
    Serial.println("[STORM ALERT] Convective signature detected on-chip! Switched to 1-minute rapid storm mode.");
  } else if (!edge_detector.in_storm_mode && edge_detector.config.interval_minutes == 15 && result.message != NULL && strstr(result.message, "Storm Over")) {
    Serial.println("[STORM OVER] Atmosphere stabilized for 15 consecutive minutes. Reverted to 15-minute synoptic routine.");
  }

  // Print Edge Diagnostic Telemetry in JSON format
  // Transmitted via:
  // 1. Routine 15-min frames: standard meteorological packet over LoRaWAN or 4G LTE-M
  // 2. Rapid 1-min storm frames: emergency high-frequency burst
  // 3. Urgent anomaly flags: immediate fault alert frame
  Serial.print("{\"step\": ");
  Serial.print(step_counter);
  Serial.print(", \"interval_m\": ");
  Serial.print(edge_detector.config.interval_minutes);
  Serial.print(", \"temp\": ");
  Serial.print(current_temp, 2);
  Serial.print(", \"press\": ");
  Serial.print(current_press, 2);
  Serial.print(", \"rh\": ");
  Serial.print(current_rh, 1);
  Serial.print(", \"dew_point\": ");
  Serial.print(result.dew_point, 2);
  Serial.print(", \"is_anomaly\": ");
  Serial.print(result.is_anomaly ? "true" : "false");
  Serial.print(", \"is_weather_event\": ");
  Serial.print(result.is_weather_event ? "true" : "false");
  Serial.print(", \"status\": \"");
  Serial.print(result.message);
  Serial.print("\", \"latency_us\": ");
  Serial.print(elapsed_us);
  Serial.println("}");

  step_counter++;
  // Demo mode: 1 second delay between steps.
  // In production field deployment:
  // uint64_t sleep_us = (uint64_t)edge_detector.config.interval_minutes * 60 * 1000000ULL;
  // esp_sleep_enable_timer_wakeup(sleep_us);
  // esp_deep_sleep_start();
  delay(1000);
}
