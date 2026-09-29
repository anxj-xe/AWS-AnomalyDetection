/**
 * @file esp32_anomaly_detector.ino
 * @brief Complete ESP32 Arduino Sketch demonstrating real-time edge screening
 *        for Automatic Weather Stations (AWS).
 * 
 * Hardware: ESP32 Dev Module (or ESP32-S3, ESP32-C3)
 * Sensors: BME280 / SHT31 / Analog RTD or simulated sensor stream
 * SIH 2026 Problem Statement: Edge AI for low-power deployment on ESP32
 */

#include <Arduino.h>
#include "esp32_anomaly_detector.h"

// Instantiate edge detector state
static aws_edge_detector_t edge_detector;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=======================================================");
  Serial.println("  SIH 2026: AWS Intelligent Edge Anomaly Detector");
  Serial.println("  Platform: ESP32 (Xtensa 32-bit @ 240MHz)");
  Serial.println("  WMO-No. 8 Standard Embedded Quality Control Active");
  Serial.println("=======================================================\n");

  aws_edge_init(&edge_detector);
}

// Function to simulate incoming sensor read (e.g. from I2C BME280)
void read_simulated_aws_sensors(float *out_t, float *out_p, float *out_rh, uint32_t step) {
  // Base diurnal cycle
  float hour = (step % 144) / 6.0; // 0 to 24 hours
  float angle = 2.0 * 3.14159 * (hour - 8.5) / 24.0;
  
  *out_t = 28.0 + 6.0 * sin(angle) + ((rand() % 20 - 10) / 100.0);
  *out_rh = 65.0 - 20.0 * sin(angle) + ((rand() % 40 - 20) / 100.0);
  *out_p = 1012.0 + 1.5 * cos(4.0 * 3.14159 * hour / 24.0);

  // Injected Faults for demonstration:
  if (step == 15) {
    // Inject sudden temperature spike (+12 C)
    *out_t += 12.0;
  } else if (step >= 25 && step <= 32) {
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

  // Measure execution latency on ESP32
  uint32_t t_start = micros();
  aws_detection_result_t result = aws_edge_process(&edge_detector, current_temp, current_press, current_rh);
  uint32_t elapsed_us = micros() - t_start;

  // Print Edge Diagnostic Telemetry in JSON format (ready for LoRaWAN / Cellular transmission)
  Serial.print("{\"step\": ");
  Serial.print(step_counter);
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
  delay(1000); // Sample every 1 second in demo mode (in real station: 1-10 mins)
}
