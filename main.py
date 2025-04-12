import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 60
ROAD_WIDTH = 80
VEHICLE_SIZE = (20, 40)
MAX_VEHICLES = 50
INTERSECTION_SIZE = 80
LANE_COUNT = 2
LIGHT_RADIUS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
VEHICLE_COLORS = [(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)) for _ in range(10)]

# Direction constants
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Input: traffic density for each direction (4) + time since red for each direction (4)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)  # Output: action values for each possible light config

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self):
        self.model = NeuralNetwork()
        self.target_model = NeuralNetwork()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_counter = 0
        
        self.loss_history = []
        self.reward_history = []
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([i[0] for i in minibatch])
        actions = torch.LongTensor([i[1] for i in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([i[2] for i in minibatch])
        next_states = torch.FloatTensor([i[3] for i in minibatch])
        dones = torch.FloatTensor([i[4] for i in minibatch])
        
        # Current Q values
        curr_q = self.model(states).gather(1, actions)
        
        # Next Q values using target model
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        
        # Target Q values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update model
        loss = self.criterion(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        
        self.loss_history.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Periodically update target network
        self.update_target_counter += 1
        if self.update_target_counter >= 100:
            self.update_target_model()
            self.update_target_counter = 0

class Vehicle:
    def __init__(self, direction):
        self.direction = direction
        self.waiting_time = 0
        self.crossed = False
        self.emissions = 0
        self.color = random.choice(VEHICLE_COLORS)
        
        # Set initial position based on direction
        mid = SCREEN_WIDTH // 2
        margin = ROAD_WIDTH * 3 // 4
        lane_offset = ROAD_WIDTH // 4
        
        if direction == NORTH:  # Coming from north
            self.x = mid - lane_offset
            self.y = 0 - VEHICLE_SIZE[1]
            self.speed = (0, 2)
        elif direction == EAST:  # Coming from east
            self.x = SCREEN_WIDTH - VEHICLE_SIZE[0]
            self.y = mid - lane_offset
            self.speed = (-2, 0)
        elif direction == SOUTH:  # Coming from south
            self.x = mid + lane_offset
            self.y = SCREEN_HEIGHT
            self.speed = (0, -2)
        else:  # Coming from west
            self.x = 0 - VEHICLE_SIZE[0]
            self.y = mid + lane_offset
            self.speed = (2, 0)
    
    def move(self, traffic_lights, vehicles):
        intersection_mid = SCREEN_WIDTH // 2
        intersection_left = intersection_mid - ROAD_WIDTH // 2
        intersection_right = intersection_mid + ROAD_WIDTH // 2
        intersection_top = SCREEN_HEIGHT // 2 - ROAD_WIDTH // 2
        intersection_bottom = SCREEN_HEIGHT // 2 + ROAD_WIDTH // 2
        
        # Check if vehicle is at the intersection
        at_intersection = False
        
        if self.direction == NORTH and self.y + VEHICLE_SIZE[1] >= intersection_top - 10 and self.y <= intersection_top:
            at_intersection = True
        elif self.direction == EAST and self.x <= intersection_right + 10 and self.x + VEHICLE_SIZE[0] >= intersection_right:
            at_intersection = True
        elif self.direction == SOUTH and self.y <= intersection_bottom + 10 and self.y + VEHICLE_SIZE[1] >= intersection_bottom:
            at_intersection = True
        elif self.direction == WEST and self.x + VEHICLE_SIZE[0] >= intersection_left - 10 and self.x <= intersection_left:
            at_intersection = True
        
        # Check if vehicle is already in the intersection
        in_intersection = (
            intersection_left <= self.x + VEHICLE_SIZE[0] // 2 <= intersection_right and
            intersection_top <= self.y + VEHICLE_SIZE[1] // 2 <= intersection_bottom
        )
        
        # Check for collision with other vehicles in front
        can_move = True
    
        for v in vehicles:
            if v != self and not v.crossed:
                # Simple collision detection
                if self.direction == NORTH:
                    if v.direction == NORTH and v.y > self.y and v.y - (self.y + VEHICLE_SIZE[1]) < 10:
                        can_move = False
                elif self.direction == EAST:
                    if v.direction == EAST and v.x < self.x and self.x - (v.x + VEHICLE_SIZE[0]) < 10:
                        can_move = False
                elif self.direction == SOUTH:
                    if v.direction == SOUTH and v.y < self.y and self.y - (v.y + VEHICLE_SIZE[1]) < 10:
                        can_move = False
                elif self.direction == WEST:
                    if v.direction == WEST and v.x > self.x and v.x - (self.x + VEHICLE_SIZE[0]) < 10:
                        can_move = False
        
        # Check if we need to stop at red light
        if at_intersection and not in_intersection:
            if (self.direction == NORTH and traffic_lights[0] == RED) or \
               (self.direction == EAST and traffic_lights[1] == RED) or \
               (self.direction == SOUTH and traffic_lights[2] == RED) or \
               (self.direction == WEST and traffic_lights[3] == RED):
                can_move = False
                self.waiting_time += 1
        
        if can_move:
            self.x += self.speed[0]
            self.y += self.speed[1]
        else:
            self.emissions += 0.2
        # Check if vehicle has crossed the screen
        if (self.direction == NORTH and self.y > SCREEN_HEIGHT) or \
           (self.direction == EAST and self.x < 0) or \
           (self.direction == SOUTH and self.y < 0) or \
           (self.direction == WEST and self.x > SCREEN_WIDTH):
            self.crossed = True

class TrafficSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Traffic Light Control with Neural Network')
        self.clock = pygame.time.Clock()
        
        self.reset()
        
        # Neural network agent
        self.agent = DQNAgent()
        
        # Training settings
        self.train_mode = True
        if not self.train_mode:
            self.agent.epsilon = 0.0
        self.episode = 0
        self.total_rewards = []
        self.avg_waiting_times = []
        self.max_episodes = 10
        
    
    def reset(self):
        # Traffic lights (NORTH, EAST, SOUTH, WEST)
        self.traffic_lights = [RED, RED, RED, RED]
        self.green_direction = random.randint(0, 3)
        self.traffic_lights[self.green_direction] = GREEN
        
        # Red light counters
        self.red_time = [0, 0, 0, 0]
        for i in range(4):
            if self.traffic_lights[i] == RED:
                self.red_time[i] = 1
        
        # Spawn new vehicles
        self.vehicles = []
        for _ in range(20):
          for direction in [1, 2, 3, 0]:
            self.vehicles.append(Vehicle(direction))
        """for _ in range(20):
            direction = random.randint(0, 3)
            self.vehicles.append(Vehicle(direction))
        """
        # Statistics
        self.time_steps = 0
        self.total_waiting_time = 0
        self.vehicles_crossed = 0
        self.current_reward = 0
    
    def get_state(self):
        # Calculate traffic density
        traffic_density = [0, 0, 0, 0]
        for vehicle in self.vehicles:
            if not vehicle.crossed:
                traffic_density[vehicle.direction] += 1
        
        # Normalize traffic density
        max_density = max(traffic_density) if max(traffic_density) > 0 else 1
        traffic_density = [d / max_density for d in traffic_density]
        
        # Normalize red light time
        max_red_time = max(self.red_time) if max(self.red_time) > 0 else 1
        red_time_norm = [t / max_red_time for t in self.red_time]
        
        # Combine features
        state = traffic_density + red_time_norm
        return state
    
    def take_action(self, action):
        # Actions: 0=North-South Green, 1=East-West Green, 2=North Green, 3=East Green
        self.traffic_lights = [RED, RED, RED, RED]
        
        if action == 0:  # North-South green
            self.traffic_lights[NORTH] = GREEN
            self.traffic_lights[SOUTH] = GREEN
        elif action == 1:  # East-West green
            self.traffic_lights[EAST] = GREEN
            self.traffic_lights[WEST] = GREEN
        elif action == 2:  # North green
            self.traffic_lights[NORTH] = GREEN
        elif action == 3:  # East green
            self.traffic_lights[EAST] = GREEN
        
        # Update red light counters
        for i in range(4):
            if self.traffic_lights[i] == RED:
                self.red_time[i] += 1
            else:
                self.red_time[i] = 0
    
    def calculate_reward(self):
        # Calculate reward based on waiting time and vehicles crossed
        total_waiting = sum(v.waiting_time for v in self.vehicles if not v.crossed)
        crossed_count = sum(1 for v in self.vehicles if v.crossed)
        new_crossed = crossed_count - self.vehicles_crossed
        
        # Reward for vehicles crossing and penalty for waiting time
        reward = new_crossed * 10 - total_waiting * 0.2 ## 0.1
        
        # Additional penalty for very long red lights
        for t in self.red_time:
            if t > 30:  # If red for too long
                reward -= (t - 30) * 0.5
        
        self.vehicles_crossed = crossed_count
        self.current_reward += reward
        return reward
    
    def spawn_vehicles(self, prob=0.1):
        if len(self.vehicles) < MAX_VEHICLES and random.random() < prob:
            direction = random.randint(0, 3)
            self.vehicles.append(Vehicle(direction))
    
    def draw(self):
        # Draw background
        self.screen.fill(WHITE)
        
        # Draw roads
        mid = SCREEN_WIDTH // 2
        pygame.draw.rect(self.screen, GRAY, (mid - ROAD_WIDTH // 2, 0, ROAD_WIDTH, SCREEN_HEIGHT))  # Vertical road
        pygame.draw.rect(self.screen, GRAY, (0, mid - ROAD_WIDTH // 2, SCREEN_WIDTH, ROAD_WIDTH))   # Horizontal road
        
        # Draw lane markings
        for i in range(0, SCREEN_HEIGHT, 30):
            pygame.draw.rect(self.screen, WHITE, (mid - 2, i, 4, 15))  # Vertical lane markers
        
        for i in range(0, SCREEN_WIDTH, 30):
            pygame.draw.rect(self.screen, WHITE, (i, mid - 2, 15, 4))  # Horizontal lane markers
        
        # Draw intersection
        pygame.draw.rect(self.screen, GRAY, (mid - ROAD_WIDTH // 2, mid - ROAD_WIDTH // 2, ROAD_WIDTH, ROAD_WIDTH))
        
        # Draw traffic lights
        light_positions = [
            (mid - ROAD_WIDTH // 2 - LIGHT_RADIUS * 2, mid - ROAD_WIDTH // 2 - LIGHT_RADIUS * 2),  # North
            (mid + ROAD_WIDTH // 2 + LIGHT_RADIUS, mid - ROAD_WIDTH // 2 - LIGHT_RADIUS * 2),      # East
            (mid + ROAD_WIDTH // 2 + LIGHT_RADIUS, mid + ROAD_WIDTH // 2 + LIGHT_RADIUS),          # South
            (mid - ROAD_WIDTH // 2 - LIGHT_RADIUS * 2, mid + ROAD_WIDTH // 2 + LIGHT_RADIUS)       # West
        ]
        
        for i, color in enumerate(self.traffic_lights):
            pygame.draw.circle(self.screen, color, light_positions[i], LIGHT_RADIUS)
        
        # Draw vehicles
        for vehicle in self.vehicles:
            if not vehicle.crossed:
                # Draw vehicle body
                if vehicle.direction in [NORTH, SOUTH]:
                    pygame.draw.rect(self.screen, vehicle.color, (vehicle.x, vehicle.y, VEHICLE_SIZE[0], VEHICLE_SIZE[1]))
                else:
                    pygame.draw.rect(self.screen, vehicle.color, (vehicle.x, vehicle.y, VEHICLE_SIZE[1], VEHICLE_SIZE[0]))
        
        # Draw statistics
        font = pygame.font.SysFont('Arial', 14)
        waiting_text = font.render(f'Total Waiting: {sum(v.waiting_time for v in self.vehicles if not v.crossed)}', True, BLACK)
        vehicles_text = font.render(f'Vehicles: {len([v for v in self.vehicles if not v.crossed])}/{len(self.vehicles)}', True, BLACK)
        crossed_text = font.render(f'Crossed: {self.vehicles_crossed}', True, BLACK)
        reward_text = font.render(f'Reward: {self.current_reward:.1f}', True, BLACK)
        episode_text = font.render(f'Episode: {self.episode}/{self.max_episodes}', True, BLACK)
        epsilon_text = font.render(f'Epsilon: {self.agent.epsilon:.2f}', True, BLACK)
        
        self.screen.blit(waiting_text, (10, 10))
        self.screen.blit(vehicles_text, (10, 30))
        self.screen.blit(crossed_text, (10, 50))
        self.screen.blit(reward_text, (10, 70))
        self.screen.blit(episode_text, (10, 90))
        self.screen.blit(epsilon_text, (10, 110))
        
        pygame.display.flip()
    
    def run(self):
        running = True
        paused = False
        
        while running and self.episode < self.max_episodes:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_t:
                        self.train_mode = not self.train_mode
                        print(f"Training mode: {self.train_mode}")
                    elif event.key == pygame.K_r:
                        self.plot_performance()
            
            if not paused:
                if len([v for v in self.vehicles if not v.crossed]) == 0 or self.time_steps >= 1000:
                    # Episode finished
                    total_emissions = sum(v.emissions for v in self.vehicles)

                    self.total_rewards.append(self.current_reward)
                    avg_wait = self.total_waiting_time / max(1, len(self.vehicles))
                    self.avg_waiting_times.append(avg_wait)
                    
                    print(f"Episode {self.episode} - Reward: {self.current_reward:.1f}, Avg Wait: {avg_wait:.1f}, Emissions: {total_emissions:.1f}")
                    
                    self.episode += 1
                    self.reset()
                    
                    # Save model every 10 episodes
                    if self.episode % 10 == 0 and self.train_mode:
                        torch.save(self.agent.model.state_dict(), f"traffic_model_{self.episode}.pth")
                    
                    # Optional: Plot performance
                    if self.episode % 50 == 0:
                        self.plot_performance()
                
                # Get current state
                state = self.get_state()
                
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                self.take_action(action)
                
                # Move vehicles and calculate statistics
                for vehicle in self.vehicles:
                    if not vehicle.crossed:
                        vehicle.move(self.traffic_lights, self.vehicles)
                        self.total_waiting_time += vehicle.waiting_time
                
                # Calculate reward
                reward = self.calculate_reward()
                
                # Get new state
                next_state = self.get_state()
                
                # Remember experience
                done = self.time_steps >= 1000
                if self.train_mode:
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Train the network
                    self.agent.replay()
                
                # Spawn new vehicles
                self.spawn_vehicles()
                
                # Clean up crossed vehicles occasionally
                if self.time_steps % 100 == 0:
                    self.vehicles = [v for v in self.vehicles if not v.crossed or v.y > -100]
                
                self.time_steps += 1
                
                # Draw everything
                self.draw()
                
                # Control game speed
                self.clock.tick(FPS)
        
        # Final plot
        self.plot_performance()
        pygame.quit()
    
    def plot_performance(self):
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.total_rewards, label='Total Reward')
        plt.title('Training Performance')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.avg_waiting_times, label='Avg Waiting Time', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Average Waiting Time')
        plt.grid(True)
        
        if len(self.agent.loss_history) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(self.agent.loss_history[-1000:], label='Loss')
            plt.title('Training Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'traffic_performance_{self.episode}.png')
        plt.close('all')

if __name__ == "__main__":
    simulation = TrafficSimulation()
    simulation.run()
