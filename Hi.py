 
import numpy as np  
import random  
  
# Game Environment  
class RunBunRunEnvironment:  
    def __init__(self, width, height):  
        self.width = width  
        self.height = height  
        self.bun_x = width // 2  
        self.bun_y = height // 2  
        self.obstacles = []  
  
    def step(self, action):  
        # Update game state based on action  
        if action == 0:  # Move left  
            self.bun_x -= 1  
        elif action == 1:  # Move right  
            self.bun_x += 1  
        elif action == 2:  # Jump  
            self.bun_y -= 10  
  
        # Check for collisions with obstacles  
        for obstacle in self.obstacles:  
            if self.bun_x == obstacle and self.bun_y == obstacle:  
                return -1, True  # Collision, game over  
  
        # Check for carrot collection  
        if self.bun_x == self.width - 1:  
            return 1, False  # Carrot collected, game won  
  
        return 0, False  # No reward, game continues  
  
# Game State Representation  
class GameState:  
    def __init__(self, environment):  
        self.environment = environment  
        self.state = np.array([environment.bun_x, environment.bun_y] + [obstacle for obstacle in environment.obstacles])  
  
# Action Space  
actions = [0, 1, 2]  # Move left, move right, jump  
  
# Reward Function  
def reward_function(state, action, next_state, done):  
    if done:  
        if next_state.environment.bun_x == next_state.environment.width - 1:  
            return 10  # Carrot collected, high reward  
        else:  
            return -10  # Collision, low reward  
    else:  
        return 0  # No reward, game continues  
  
# AI Algorithm (Q-Learning example)  
class QLearningAgent:  
    def __init__(self, environment, alpha, epsilon, gamma):  
        self.environment = environment  
        self.alpha = alpha  
        self.epsilon = epsilon  
        self.gamma = gamma  
        self.q_values = {}  
  
    def get_q_value(self, state, action):  
        if (state, action) not in self.q_values:  
            self.q_values[(state, action)] = 0  
        return self.q_values[(state, action)]  
  
    def update_q_value(self, state, action, next_state, reward, done):  
        q_value = self.get_q_value(state, action)  
        next_q_value = max([self.get_q_value(next_state, a) for a in actions])  
        self.q_values[(state, action)] = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)  
  
    def choose_action(self, state):  
        if random.random() &lt; self.epsilon:  
            return random.choice(actions)  
        else:  
            return max(actions, key=lambda a: self.get_q_value(state, a))  
  
# Initialize environment and AI  
environment = RunBunRunEnvironment(10, 10)  
agent = QLearningAgent(environment, alpha=0.1, epsilon=0.1, gamma=0.9)  
  
# Train the AI  
for episode in range(1000):  
    state = GameState(environment)  
    done = False  
    while not done:  
        action = agent.choose_action(state.state)  
        next_state, reward, done = environment.step(action)  
        agent.update_q_value(state.state, action, next_state.state, reward, done)  
        state = next_state  
  
    print(f"Episode {episode+1} completed")  
