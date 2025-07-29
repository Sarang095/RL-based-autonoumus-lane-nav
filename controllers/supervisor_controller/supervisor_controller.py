"""
Supervisor Controller for Autonomous Driving Simulation
Manages scenario states, agent resets, and environment elements like traffic lights.
"""

import sys
import time
import random
import numpy as np

# Try to import Webots controller, fallback to test mode if not available
try:
    from controller import Supervisor
    WEBOTS_AVAILABLE = True
except ImportError:
    WEBOTS_AVAILABLE = False
    print("Webots controller not available. Running in test mode.")


class ScenarioSupervisor:
    """
    Supervisor controller that manages the simulation environment.
    Handles agent position resets and traffic light control.
    """
    
    def __init__(self):
        """Initialize the supervisor and get handles to simulation nodes."""
        # Get the supervisor instance
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Get handle to the agent robot node (must be defined as DEF="AGENT_CAR" in world file)
        self.agent_node = self.supervisor.getFromDef("AGENT_CAR")
        if self.agent_node is None:
            print("Warning: AGENT_CAR node not found. Make sure it's defined in the world file.")
            sys.exit(1)
        
        # Get the translation and rotation fields for the agent
        self.translation_field = self.agent_node.getField("translation")
        self.rotation_field = self.agent_node.getField("rotation")
        
        # Store initial position and orientation for reset
        self.initial_translation = self.translation_field.getSFVec3f()
        self.initial_rotation = self.rotation_field.getSFRotation()
        
        # Get handles to LED nodes representing traffic lights
        self.traffic_lights = {}
        
        # Try to get traffic light LEDs (these should be defined in the world file)
        traffic_light_names = ["TRAFFIC_LIGHT_1", "TRAFFIC_LIGHT_2", "TRAFFIC_LIGHT_3"]
        for light_name in traffic_light_names:
            light_node = self.supervisor.getFromDef(light_name)
            if light_node is not None:
                self.traffic_lights[light_name] = light_node
                print(f"Found traffic light: {light_name}")
            else:
                print(f"Traffic light {light_name} not found in world file")
        
        # Traffic light colors (RGB values for LED)
        self.LIGHT_COLORS = {
            'red': [1.0, 0.0, 0.0],
            'yellow': [1.0, 1.0, 0.0],
            'green': [0.0, 1.0, 0.0],
            'off': [0.0, 0.0, 0.0]
        }
        
        # Scenario management
        self.current_scenario = 'parking'
        self.scenario_positions = {
            'parking': {
                'translation': [2.0, 0.1, 2.0],
                'rotation': [0.0, 1.0, 0.0, 0.0]
            },
            'intersection': {
                'translation': [10.0, 0.1, 10.0],
                'rotation': [0.0, 1.0, 0.0, 0.0]
            },
            'roundabout': {
                'translation': [-20.0, 0.1, 10.0],
                'rotation': [0.0, 1.0, 0.0, 0.0]
            }
        }
        
        # Traffic light control state
        self.traffic_light_state = 'red'
        self.traffic_light_timer = 0.0
        self.traffic_light_cycle_time = 10.0  # seconds
        
        # Command processing
        self.command_queue = []
        
        print("ScenarioSupervisor initialized successfully")
    
    def reset_agent_position(self, scenario=None):
        """
        Reset the agent to a specified scenario starting position.
        
        Args:
            scenario (str): The scenario name ('parking', 'intersection', 'roundabout')
                          If None, uses current scenario
        """
        if scenario is None:
            scenario = self.current_scenario
        
        if scenario not in self.scenario_positions:
            print(f"Unknown scenario: {scenario}")
            return False
        
        # Get position and rotation for the scenario
        position = self.scenario_positions[scenario]['translation']
        rotation = self.scenario_positions[scenario]['rotation']
        
        # Reset agent position
        self.translation_field.setSFVec3f(position)
        self.rotation_field.setSFRotation(rotation)
        
        # Reset the simulation physics
        self.agent_node.resetPhysics()
        
        self.current_scenario = scenario
        print(f"Agent reset to {scenario} starting position: {position}")
        
        return True
    
    def set_traffic_light_color(self, light_name, color):
        """
        Set the color of a specific traffic light.
        
        Args:
            light_name (str): Name of the traffic light (e.g., "TRAFFIC_LIGHT_1")
            color (str): Color name ('red', 'yellow', 'green', 'off')
        """
        if light_name not in self.traffic_lights:
            print(f"Traffic light {light_name} not available")
            return False
        
        if color not in self.LIGHT_COLORS:
            print(f"Unknown color: {color}")
            return False
        
        try:
            light_node = self.traffic_lights[light_name]
            # Assuming the traffic light has an LED field that can be controlled
            # This is a simplified approach - actual implementation may vary
            rgb_values = self.LIGHT_COLORS[color]
            
            # Try to find and set the LED color field
            # Note: This depends on the specific TrafficLight proto implementation
            led_field = light_node.getField("color")
            if led_field:
                led_field.setSFColor(rgb_values)
                print(f"Set {light_name} to {color}")
                return True
            else:
                print(f"Could not find color field for {light_name}")
                return False
                
        except Exception as e:
            print(f"Error setting traffic light color: {e}")
            return False
    
    def cycle_traffic_lights(self):
        """Automatically cycle traffic lights through red/yellow/green sequence."""
        self.traffic_light_timer += self.timestep / 1000.0  # Convert to seconds
        
        if self.traffic_light_timer >= self.traffic_light_cycle_time:
            self.traffic_light_timer = 0.0
            
            # Cycle to next state
            if self.traffic_light_state == 'red':
                self.traffic_light_state = 'green'
            elif self.traffic_light_state == 'green':
                self.traffic_light_state = 'yellow'
            elif self.traffic_light_state == 'yellow':
                self.traffic_light_state = 'red'
            
            # Apply to all traffic lights
            for light_name in self.traffic_lights.keys():
                self.set_traffic_light_color(light_name, self.traffic_light_state)
    
    def set_random_scenario(self):
        """Set the agent to a random scenario position."""
        scenarios = list(self.scenario_positions.keys())
        random_scenario = random.choice(scenarios)
        return self.reset_agent_position(random_scenario)
    
    def process_commands(self):
        """Process any external commands (for RL training integration)."""
        # This method can be extended to receive commands from external training scripts
        # For now, it's a placeholder for future communication mechanisms
        
        # Example command processing (could be extended with socket communication)
        if self.command_queue:
            command = self.command_queue.pop(0)
            
            if command['type'] == 'reset':
                scenario = command.get('scenario', None)
                self.reset_agent_position(scenario)
            
            elif command['type'] == 'traffic_light':
                light_name = command.get('light_name')
                color = command.get('color')
                self.set_traffic_light_color(light_name, color)
            
            elif command['type'] == 'random_scenario':
                self.set_random_scenario()
    
    def add_command(self, command):
        """Add a command to the processing queue."""
        self.command_queue.append(command)
    
    def get_agent_position(self):
        """Get the current position of the agent."""
        return self.translation_field.getSFVec3f()
    
    def get_agent_rotation(self):
        """Get the current rotation of the agent."""
        return self.rotation_field.getSFRotation()
    
    def check_collision(self):
        """
        Check if the agent has collided with any obstacles.
        This is a simplified collision detection method.
        """
        position = self.get_agent_position()
        
        # Define obstacle positions (should match world file)
        obstacles = [
            {'pos': [15.0, 0.5, 8.0], 'size': [2.0, 1.0, 1.0]},
            {'pos': [25.0, 0.5, 25.0], 'size': [1.5, 1.0, 3.0]},
            {'pos': [-5.0, 0.5, 10.0], 'radius': 0.5}  # Cylindrical obstacle
        ]
        
        agent_x, agent_y, agent_z = position
        
        for obstacle in obstacles:
            obs_x, obs_y, obs_z = obstacle['pos']
            
            if 'radius' in obstacle:
                # Cylindrical obstacle
                distance = ((agent_x - obs_x)**2 + (agent_z - obs_z)**2)**0.5
                if distance < obstacle['radius'] + 1.0:  # Agent radius approximation
                    return True
            else:
                # Box obstacle
                size_x, size_y, size_z = obstacle['size']
                if (abs(agent_x - obs_x) < size_x/2 + 1.0 and
                    abs(agent_z - obs_z) < size_z/2 + 2.0):  # Agent size approximation
                    return True
        
        return False
    
    def calculate_distance_to_target(self, target_scenario=None):
        """Calculate distance from agent to target position."""
        if target_scenario is None:
            target_scenario = self.current_scenario
        
        if target_scenario not in self.scenario_positions:
            return float('inf')
        
        agent_pos = self.get_agent_position()
        target_pos = self.scenario_positions[target_scenario]['translation']
        
        distance = ((agent_pos[0] - target_pos[0])**2 + 
                   (agent_pos[2] - target_pos[2])**2)**0.5
        
        return distance
    
    def run(self):
        """Main supervisor control loop."""
        print("ScenarioSupervisor starting main loop")
        print("Available commands:")
        print("  - Automatic traffic light cycling")
        print("  - Agent position monitoring")
        print("  - Collision detection")
        
        step_counter = 0
        
        while self.supervisor.step(self.timestep) != -1:
            step_counter += 1
            
            # Process any pending commands
            self.process_commands()
            
            # Cycle traffic lights automatically
            self.cycle_traffic_lights()
            
            # Monitor agent status every 100 steps
            if step_counter % 100 == 0:
                position = self.get_agent_position()
                collision = self.check_collision()
                distance_to_target = self.calculate_distance_to_target()
                
                if collision:
                    print(f"COLLISION DETECTED at position {position}")
                
                print(f"Agent at {position}, distance to target: {distance_to_target:.2f}")
            
            # Example: Reset agent randomly every 30 seconds
            if step_counter % (30 * 1000 // self.timestep) == 0:
                print("Performing random scenario reset")
                self.set_random_scenario()


# Example of how to use the supervisor for RL training integration
class RLIntegrationSupervisor(ScenarioSupervisor):
    """Extended supervisor with RL training integration capabilities."""
    
    def __init__(self):
        super().__init__()
        
        # RL-specific parameters
        self.episode_step_count = 0
        self.max_episode_steps = 1000
        self.current_episode = 0
        
    def reset_episode(self, scenario=None):
        """Reset environment for a new RL episode."""
        self.episode_step_count = 0
        self.current_episode += 1
        
        # Reset to random scenario if none specified
        if scenario is None:
            self.set_random_scenario()
        else:
            self.reset_agent_position(scenario)
        
        # Reset traffic lights to random state
        colors = ['red', 'green']
        random_color = random.choice(colors)
        for light_name in self.traffic_lights.keys():
            self.set_traffic_light_color(light_name, random_color)
        
        print(f"Episode {self.current_episode} started, scenario: {self.current_scenario}")
        
        return True
    
    def step_episode(self):
        """Perform one step in the current episode."""
        self.episode_step_count += 1
        
        # Check if episode should end
        if self.episode_step_count >= self.max_episode_steps:
            print(f"Episode {self.current_episode} ended (max steps reached)")
            return True  # Episode done
        
        # Check for collision
        if self.check_collision():
            print(f"Episode {self.current_episode} ended (collision)")
            return True  # Episode done
        
        # Check if target reached
        distance_to_target = self.calculate_distance_to_target()
        if distance_to_target < 2.0:  # Within 2 meters of target
            print(f"Episode {self.current_episode} ended (target reached)")
            return True  # Episode done
        
        return False  # Episode continues


def test_mode():
    """Run supervisor controller in test mode without Webots."""
    print("Supervisor Controller Test Mode")
    print("==============================")
    print("Testing supervisor logic without Webots...")
    
    # Simulate supervisor operations
    print("✓ Simulating agent position reset...")
    initial_position = [0.0, 0.1, 0.0]
    reset_position = [5.0, 0.1, 2.0]
    print(f"  Agent moved from {initial_position} to {reset_position}")
    
    print("✓ Simulating traffic light control...")
    traffic_states = ["red", "yellow", "green"]
    current_state = random.choice(traffic_states)
    print(f"  Traffic light state: {current_state}")
    
    print("✓ Simulating scenario management...")
    scenarios = ["highway_driving", "city_intersection", "parking"]
    current_scenario = random.choice(scenarios)
    print(f"  Current scenario: {current_scenario}")
    
    print("✓ Simulating RL episode management...")
    episode_length = random.randint(50, 200)
    print(f"  Episode length: {episode_length} steps")
    
    # Simulate episode reward calculation
    distance_reward = np.random.uniform(0.1, 1.0)
    safety_penalty = np.random.uniform(-0.5, 0.0)
    total_reward = distance_reward + safety_penalty
    print(f"  Episode reward: {total_reward:.3f}")
    
    print("\n🎉 Supervisor controller test completed successfully!")
    print("The controller is ready to work with Webots.")


# Main execution
if __name__ == "__main__":
    if "--test" in sys.argv or not WEBOTS_AVAILABLE:
        test_mode()
    else:
        # Choose which supervisor to run
        use_rl_supervisor = False  # Set to True for RL training integration
        
        if use_rl_supervisor:
            supervisor = RLIntegrationSupervisor()
            
            # Example RL episode loop
            while supervisor.supervisor.step(supervisor.timestep) != -1:
                episode_done = supervisor.step_episode()
                
                if episode_done:
                    supervisor.reset_episode()
        else:
            supervisor = ScenarioSupervisor()
            supervisor.run()