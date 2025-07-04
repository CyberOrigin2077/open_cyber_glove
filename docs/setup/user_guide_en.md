# Get Started Guide

## Potential Risks and Limitations

**1. Due to limitations in the durability of the fabric material, we are unable to provide a long-term warranty comparable to that of mechanical products. Instead, we offer a limited 3-month replacement policy at no additional cost.**

**2. Given the high sensitivity of the flexible sensors within the gloves to individual hand morphology, there may be instances where the pre-provided models do not fully meet the user's precision requirements.**

## Unbox

### Packing List

| Item              | Quantity           |
| :---------------- | :----------------- |
| Storage Case      | 1                  |
| OpenCyberGlove    | 2 (left and right) |
| USB Type-C Cable  | 2                  |
| Quick Start Guide | 1                  |

### Parameter

Parameter for a single glove:

| Item               | Detail     |
| :----------------- | :--------- |
| Sensor Number      | 19         |
| Weight             | 25 g       |
| Size (Medium)      | 17 by 10 cm|
| Hardware Interface | USB Type-C |
| Communication      | UART       |
| Frequency          | 120 Hz     |
| Sensor Value Range | 0 - 16380  |
| Power              | ~0.3 w     |

### Appearance

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="../../imgs/userguide_product_diagram1_en.png" width="400" />
  <img src="../../imgs/userguide_product_diagram2_en.png" width="400" />
  
</div>

## Wear

Follow the steps below to wear the glove to ensure sensor accuracy.

Before wearing, please refer to the size chart below to ensure that your hand size has a good fit for the glove. Currently, only medium size (M) is available.

| Size                    | Palm Circ. (mm) | Wrist Circ. (mm) | Hand Length (mm) | Middle Finger (mm) | Index Finger (mm) | Thumb (mm) | Ring Finger (mm) | Little Finger (mm) |
| :---------------------- | :-------------- | :--------------- | :--------------- | :----------------- | :---------------- | :--------- | :--------------- | :----------------- |
| Medium Glove Size Range | 80-105          | 155-180          | 180-195          | 77-90              | 67-80             | 55-69      | 67-80            | 50-65              |
| Recommended Best Fit    | 89.2            | 165.9            | 186.6            | 83.7               | 72.3              | 62.5       | 74.8             | 60.0               |

* **Finger Alignment**: Put on the glove, ensuring fingers are fully inserted into their respective slots. The webbings between the fingers must align with the base of the glove's finger seams. The fingertips should align with the stitched seam at the tip of the glove.
* **Surface Smoothing**: Smooth the surface of the glove to ensure there are no twists or wrinkles in the sensor area.
* **Wrist Fastening**: Tighten the adjustment strap to secure the glove, while ensuring freedom of wrist movement.

<div style="display: flex; justify-content: center; align-items: center; gap: 1;">
  <img src="../../imgs/userguide_wearing1_en.png" width="300" />
  <img src="../../imgs/userguide_wearing2_en.png" width="300" />
</div>
<div style="display: flex; justify-content: center; align-items: center">
  <img src="../../imgs/userguide_wearing.gif" width="343" style="transform: rotate(-90deg);"/>
</div>

## Connect

Follow the steps below to connect the glove correctly:

* Use the provided dedicated data cable. The glove is powered and transmits data via the data cable; no external power source is required.
* Connect one end of the data cable to the glove's USB-C data port and the other end to the computer's USB port. Observe the status indicator light to determine the glove's status.

Status Indicator Light Guide:

* Light Off: Not powered on or improperly connected.
* Solid Green: Powered on, awaiting connection.
* Breathing Blue Light: Connected, in standby mode.

## Calibrate

**Note: Since flex sensor is highly sensitive to extension, calibration can significantly affect the accuracy of model-based joint estimation methods.**

When using the glove SDK, please follow the on-screen prompts to perform the specified gestures to fully relax or activate the sensors. Recalibration is required when changing users or if a decrease in accuracy is perceived.

Gesture examples are as follows:

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="../../imgs/userguide_calibration1_en.png" width="300" />
  <img src="../../imgs/userguide_calibration2_en.png" width="300" />
</div>
<div style="display: flex; justify-content: center; align-items: center; gap: 143px;">
  <img src="../../imgs/userguide_calibration1.gif" width="170" style="transform: rotate(-90deg);" />
  <img src="../../imgs/userguide_calibration2.gif" width="170" style="transform: rotate(-90deg);" />
</div>
