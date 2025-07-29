# Manual Driving Simulator - Setup Guide

## Project Overview

This Webots project provides a complete manual driving simulation environment where users can practice driving skills in complex scenarios including parking lots, roundabouts, and intersections. The simulation uses **keyboard control only** - no external programming languages are required.

## Project Structure

```
manual_driving_simulator/
├── protos/
│   ├── SmartCar.proto          # Keyboard-controlled vehicle with sensors
│   ├── ParkingLot.proto        # Parking area with painted spaces
│   ├── Roundabout.proto        # Multi-exit roundabout
│   └── TrafficLight.proto      # Traffic lights with LED nodes
├── worlds/
│   └── driving_world.wbt       # Main simulation world
├── manual_driving_simulator.wbproj
└── MANUAL_DRIVING_SETUP.md     # This file
```

## System Requirements

- **Webots R2023b** or later
- Minimum 4GB RAM
- Graphics card with OpenGL 3.3 support
- Keyboard for vehicle control

## Installation and Setup

### Step 1: Install Webots
1. Download Webots from: https://cyberbotics.com
2. Install following the platform-specific instructions
3. Launch Webots to verify installation

### Step 2: Open the Project
1. Launch Webots
2. Go to `File > Open Project...`
3. Navigate to the project directory and select `manual_driving_simulator.wbproj`
4. The project will open with the default world

### Step 3: Load the Driving World
1. In Webots, go to `File > Open World...`
2. Navigate to `worlds/driving_world.wbt`
3. Click "Open"
4. The simulation should load with the SmartCar and complete environment

### Step 4: Start the Simulation
1. Click the "Play" button (▶️) in the toolbar
2. The simulation will start and the car should be ready for keyboard control
3. Use the keyboard controls (see below) to drive the vehicle

## Keyboard Controls

The SmartCar uses Webots' built-in keyboard controller. The following keys control the vehicle:

### Movement Controls
- **↑ (Up Arrow)**: Accelerate forward
- **↓ (Down Arrow)**: Brake / Reverse
- **← (Left Arrow)**: Turn left
- **→ (Right Arrow)**: Turn right

### Additional Controls (if available in your Webots version)
- **Shift + Arrow Keys**: More precise steering
- **Spacebar**: Handbrake (if implemented)

## Simulation Features

### 1. SmartCar Vehicle
- **Sensors**: Camera, Distance sensors (front, rear, left, right), IMU, GPS
- **Control**: Keyboard-based manual control
- **Physics**: Realistic wheel physics and dynamics
- **Visual**: Red sports car appearance

### 2. Parking Lot
- **Features**: 14 painted parking spaces (7 per row)
- **Layout**: Two rows with driving lane in center
- **Markings**: White painted lines and directional arrows
- **Practice**: Learn parallel parking and backing maneuvers

### 3. Roundabout
- **Design**: 4-exit roundabout with proper lane markings
- **Features**: Center island, yield markings, directional arrows
- **Practice**: Navigate multi-lane roundabout entry/exit
- **Realism**: Right-hand traffic flow patterns

### 4. Intersection
- **Layout**: 4-way intersection with traffic lights
- **Traffic Lights**: Static red lights (always on)
- **Markings**: Stop lines and lane dividers
- **Practice**: Stopping, turning, and intersection navigation

### 5. Road Network
- **North-South Road**: 80m long main thoroughfare
- **East-West Road**: 80m long cross street
- **Connecting Roads**: Links between all major areas
- **Surface**: Realistic asphalt texture and markings

## Driving Practice Scenarios

### Scenario 1: Basic Driving
1. Start at the southern position
2. Drive north on the main road
3. Practice staying in lanes and following road markings
4. Stop at the intersection

### Scenario 2: Parking Practice
1. Navigate to the parking lot (west side)
2. Enter through the marked entrance
3. Practice parking in different spaces
4. Try both forward and reverse parking

### Scenario 3: Roundabout Navigation
1. Drive to the roundabout (northeast area)
2. Enter the roundabout using proper lane
3. Practice exiting at different points
4. Follow the directional flow arrows

### Scenario 4: Complex Navigation
1. Start at parking lot
2. Navigate to roundabout via connecting roads
3. Exit roundabout toward intersection
4. Stop at traffic lights
5. Complete full circuit back to parking

## Troubleshooting

### Car Doesn't Move
- Ensure simulation is running (Play button pressed)
- Check that SmartCar is selected in scene tree
- Verify keyboard focus is on Webots window

### Poor Performance
- Reduce graphics quality in Webots preferences
- Close other applications to free memory
- Check graphics driver updates

### Missing Visual Elements
- Reload the world file (Ctrl+Shift+R)
- Verify all PROTO files are in protos/ directory
- Check Webots console for error messages

### Controller Issues
- Car should use built-in `<keyboard>` controller
- No external controllers needed
- Reset simulation if keyboard becomes unresponsive

## Advanced Features

### Camera View
- The SmartCar includes a front-facing camera
- Access camera view through robot devices panel
- Useful for first-person driving perspective

### Sensor Data
- Distance sensors provide parking assistance
- IMU shows vehicle orientation
- GPS provides position information
- Data visible in robot console

### Customization
- Modify car color in SmartCar.proto
- Adjust traffic light states in TrafficLight.proto
- Change parking lot size in ParkingLot.proto
- Modify roundabout radius in Roundabout.proto

## Tips for Effective Practice

1. **Start Slow**: Begin with gentle acceleration and steering
2. **Use References**: Follow lane markings and road signs
3. **Practice Scenarios**: Focus on one skill at a time
4. **Check Surroundings**: Use distance sensors for parking
5. **Realistic Approach**: Drive as you would in real life

## File Descriptions

### PROTO Files
- **SmartCar.proto**: Complete vehicle definition with sensors and keyboard control
- **ParkingLot.proto**: Reusable parking facility with painted spaces
- **Roundabout.proto**: Multi-exit circular intersection
- **TrafficLight.proto**: Intersection traffic control with LED lights

### World File
- **driving_world.wbt**: Complete simulation environment assembling all elements

## Support and Modifications

This project is designed to be self-contained and requires no external programming. All functionality is built into the PROTO files and world definition.

For modifications:
1. Edit PROTO files to change component behavior
2. Modify world file to adjust layout
3. All changes take effect after reloading the simulation

## Version Information

- **Webots Version**: R2023b+
- **Project Version**: 1.0
- **Last Updated**: 2024
- **License**: Educational Use

---

**Enjoy practicing your driving skills in this comprehensive simulation environment!**