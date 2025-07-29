# Manual Driving Simulation - Webots Project

## Overview

This is a complete Webots manual driving simulation project that features a keyboard-controlled vehicle navigating through complex scenarios including:

- **Parking Lot**: A structured parking area with numbered spaces and clear markings
- **Roundabout**: A multi-exit roundabout with yield signs and proper lane markings
- **Intersection**: A multi-way intersection with traffic lights
- **Road Network**: Connected roads linking all scenarios

## Project Structure

```
manual_driving_simulation/
├── worlds/
│   └── driving_world.wbt          # Main world file
├── protos/
│   ├── SmartCar.proto            # Vehicle with full sensor suite
│   ├── ParkingLot.proto          # Parking lot with numbered spaces
│   ├── Roundabout.proto          # Multi-exit roundabout
│   └── TrafficLight.proto        # Traffic lights with LED nodes
└── README.md                     # This file
```

## Vehicle Features

The SmartCar includes:
- **Camera**: Front-facing camera for vision
- **Distance Sensors**: 8 sensors for parking assistance
  - Front, rear, left, right sensors
  - Corner sensors (front-left, front-right, rear-left, rear-right)
- **Inertial Unit**: For orientation and motion sensing
- **GPS**: For position tracking
- **Headlights**: Automatic illumination
- **Keyboard Control**: Direct manual control

## Setup Instructions

### Step 1: Project Folder Setup
1. Create a new directory for your Webots project
2. Copy all files maintaining the folder structure:
   - `worlds/driving_world.wbt`
   - `protos/SmartCar.proto`
   - `protos/ParkingLot.proto`
   - `protos/Roundabout.proto`
   - `protos/TrafficLight.proto`

### Step 2: Open in Webots
1. Launch Webots
2. Go to **File → Open World**
3. Navigate to your project folder
4. Select `worlds/driving_world.wbt`
5. Click **Open**

### Step 3: Start Simulation
1. Click the **Play** button in Webots toolbar
2. The simulation will start with the car positioned on the main road
3. The camera will follow the car automatically

## Keyboard Controls

### Basic Movement
- **W** or **↑**: Accelerate forward
- **S** or **↓**: Accelerate backward (reverse)
- **A** or **←**: Steer left
- **D** or **→**: Steer right

### Additional Controls
- **Space**: Brake/Stop
- **Shift**: Increase acceleration (turbo)
- **Ctrl**: Decrease acceleration (slow driving)

### Camera Controls (while simulation is running)
- **Mouse**: Look around
- **Mouse Wheel**: Zoom in/out
- **Right Click + Drag**: Rotate view
- **Middle Click + Drag**: Pan view

## Driving Scenarios

### 1. Parking Practice
- **Location**: East side of the world (right side)
- **Features**: 24 numbered parking spaces in 3 rows
- **Practice**: Parallel parking, perpendicular parking, tight space maneuvering
- **Sensors**: Use distance sensors to avoid collisions

### 2. Roundabout Navigation
- **Location**: Northwest area of the world
- **Features**: Multi-exit roundabout with yield signs
- **Practice**: Proper roundabout entry/exit, yielding to traffic
- **Rules**: Yield to vehicles already in the roundabout

### 3. Intersection Management
- **Location**: Center of the world
- **Features**: 4-way intersection with traffic lights
- **Practice**: Traffic light compliance, turning maneuvers
- **Lights**: Red lights are on by default (static)

### 4. Road Network
- **Features**: Connected roads between all scenarios
- **Practice**: Lane keeping, smooth transitions between areas

## Tips for Driving

### Parking
1. Use distance sensors to gauge proximity to obstacles
2. Practice both forward and reverse parking
3. Try different parking spaces with varying difficulty

### Roundabout
1. Approach at reduced speed
2. Yield to vehicles already in the roundabout
3. Signal your intended exit
4. Maintain proper lane position

### Intersection
1. Come to complete stops at red lights
2. Check all directions before proceeding
3. Practice left and right turns

### General Driving
1. Maintain reasonable speeds
2. Use smooth steering inputs
3. Practice emergency braking
4. Get familiar with the car's dimensions using sensors

## Sensor Information

### Distance Sensor Readings
- Values range from 0 to 2000 (2 meters maximum range)
- Lower values indicate closer obstacles
- Use for parking assistance and collision avoidance

### Camera Feed
- 640x480 resolution
- Wide field of view for good visibility
- Positioned at realistic height on vehicle

### GPS Coordinates
- Provides real-time position data
- Useful for navigation and position tracking

## Troubleshooting

### Car Not Moving
- Ensure the simulation is running (Play button pressed)
- Check that the car controller is set to "<keyboard>"
- Try pressing W or arrow keys firmly

### Poor Performance
- Close unnecessary applications
- Reduce Webots graphics quality in Preferences
- Ensure your system meets Webots requirements

### PROTO Files Not Loading
- Verify all PROTO files are in the `protos/` folder
- Check that file paths in the world file are correct
- Restart Webots if changes were made to PROTO files

## Advanced Features

### Multiple Camera Views
- Right-click on the car and select "Follow Object" for different camera modes
- Try "Mounted Shot", "Pan and Tilt", or "Static" views

### Sensor Visualization
- Enable sensor ray visualization in Webots view menu
- Helpful for understanding sensor coverage

### Physics Tuning
- Vehicle mass: 1500kg (realistic car weight)
- Wheel friction: Optimized for road surfaces
- Suspension: Simulated through wheel physics

## Educational Objectives

This simulation helps practice:
1. **Vehicle Control**: Mastering acceleration, braking, and steering
2. **Spatial Awareness**: Understanding vehicle dimensions and positioning
3. **Traffic Rules**: Following proper traffic protocols
4. **Parking Skills**: Various parking scenarios and techniques
5. **Sensor Usage**: Learning to rely on distance sensors for assistance

## Technical Notes

- All functionality is self-contained within Webots
- No external programming required
- Uses built-in keyboard controller
- Compatible with Webots R2023b and later
- PROTO files are modular and reusable

Enjoy practicing your driving skills in this comprehensive simulation environment!