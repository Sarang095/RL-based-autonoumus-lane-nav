"""
Supervisor Controller for Autonomous Driving Simulation
Manages scenario states, agent resets, and environment elements like traffic lights.
"""

from controller import Supervisor
import struct
import sys


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
        self.colors = {
            'red': [1.0, 0.0, 0.0],
            'yellow': [1.0, 1.0, 0.0],
            'green': [0.0, 1.0, 0.0],
            'off': [0.0, 0.0, 0.0]
        }
        
        # Communication receiver for commands from training script
        self.receiver = self.supervisor.getDevice("receiver")
        if self.receiver:
            self.receiver.enable(self.timestep)
        
        # Communication emitter to send data back to training script
        self.emitter = self.supervisor.getDevice("emitter")
        
        print("ScenarioSupervisor initialized successfully")
    
    def reset_agent_position(self):
        """
        Reset the agent car to its initial position and orientation.
        This is called at the beginning of each training episode.
        """
        # Reset position
        self.translation_field.setSFVec3f(self.initial_translation)
        
        # Reset orientation
        self.rotation_field.setSFRotation(self.initial_rotation)
        
        # Restart the agent controller to reset its internal state
        self.agent_node.restartController()
        
        print("Agent position reset to initial state")
    
    def set_traffic_light_color(self, light_name, color):
        """
        Set the color of a specific traffic light.
        
        Args:
            light_name (str): Name of the traffic light (e.g., "TRAFFIC_LIGHT_1")
            color (str): Color name ('red', 'yellow', 'green', 'off')
        """
        if light_name in self.traffic_lights and color in self.colors:
            # Get the LED field and set its color
            led_field = self.traffic_lights[light_name].getField("color")
            if led_field:
                led_field.setSFColor(self.colors[color])
                print(f"Set {light_name} to {color}")
            else:
                print(f"Warning: Could not find LED color field for {light_name}")
        else:
            print(f"Warning: Invalid light name '{light_name}' or color '{color}'")
    
    def set_all_traffic_lights(self, color):
        """
        Set all traffic lights to the same color.
        
        Args:
            color (str): Color name ('red', 'yellow', 'green', 'off')
        """
        for light_name in self.traffic_lights.keys():
            self.set_traffic_light_color(light_name, color)
    
    def get_agent_position(self):
        """
        Get the current position of the agent.
        
        Returns:
            list: [x, y, z] coordinates of the agent
        """
        return self.translation_field.getSFVec3f()
    
    def get_agent_rotation(self):
        """
        Get the current rotation of the agent.
        
        Returns:
            list: [x, y, z, angle] rotation of the agent
        """
        return self.rotation_field.getSFRotation()
    
    def process_commands(self):
        """
        Process commands received from the training script.
        Commands are sent as binary data.
        """
        if self.receiver and self.receiver.getQueueLength() > 0:
            # Receive the message
            message = self.receiver.getData()
            
            try:
                # Unpack the command (assuming it's a string)
                command = message.decode('utf-8').strip()
                
                if command == "reset":
                    self.reset_agent_position()
                    # Send acknowledgment
                    if self.emitter:
                        self.emitter.send("reset_done".encode('utf-8'))
                
                elif command.startswith("light_"):
                    # Parse light command: "light_LIGHT_NAME_COLOR"
                    parts = command.split('_')
                    if len(parts) >= 3:
                        light_name = '_'.join(parts[1:-1])
                        color = parts[-1]
                        self.set_traffic_light_color(light_name, color)
                
                elif command == "lights_red":
                    self.set_all_traffic_lights('red')
                elif command == "lights_green":
                    self.set_all_traffic_lights('green')
                elif command == "lights_yellow":
                    self.set_all_traffic_lights('yellow')
                elif command == "lights_off":
                    self.set_all_traffic_lights('off')
                
            except Exception as e:
                print(f"Error processing command: {e}")
            
            # Clear the message
            self.receiver.nextPacket()
    
    def run(self):
        """
        Main execution loop for the supervisor.
        Processes commands and maintains the simulation.
        """
        print("Starting supervisor main loop...")
        
        # Initialize traffic lights to red
        self.set_all_traffic_lights('red')
        
        while self.supervisor.step(self.timestep) != -1:
            # Process any incoming commands
            self.process_commands()
            
            # Additional scenario-specific logic can be added here
            # For example, automatic traffic light cycling, dynamic obstacle placement, etc.


def main():
    """Main function to run the supervisor controller."""
    supervisor = ScenarioSupervisor()
    supervisor.run()


if __name__ == "__main__":
    main()