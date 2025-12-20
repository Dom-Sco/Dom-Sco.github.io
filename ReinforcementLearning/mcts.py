import math
import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> Node
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = state.get_legal_actions()  # depends on game

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c=1.4):
        # Select child with highest UCT
        def uct_value(child):
            if child.visits == 0:
                return float('inf')  # Explore unvisited child first
            exploit = child.total_reward / child.visits
            explore = math.sqrt(math.log(self.visits) / child.visits)
            return exploit + c * explore

        return max(self.children.values(), key=uct_value)

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = Node(next_state, parent=self)
        self.children[action] = child_node
        return child_node

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward


def rollout_policy(state):
    # Default: random rollout
    while not state.is_terminal():
        action = random.choice(state.get_legal_actions())
        state = state.move(action)
    return state.get_result()  # e.g., -1, 0, 1


def mcts(root_state, iterations=1000):
    root = Node(root_state)

    # --- Initial exploration of all actions at the root ---
    while root.untried_actions:
        node = root.expand()
        reward = rollout_policy(node.state)
        while node is not None:
            node.update(reward)
            node = node.parent
    # -------------------------------------------------------

    for _ in range(iterations):
        node = root

        # 1. Selection
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child()

        # 2. Expansion
        if not node.state.is_terminal():
            node = node.expand()

        # 3. Simulation (Rollout)
        reward = rollout_policy(node.state)

        # 4. Backpropagation
        while node is not None:
            node.update(reward)
            node = node.parent

    # Final action selection (e.g., most visited child)
    return max(root.children.items(), key=lambda item: item[1].visits)[0]