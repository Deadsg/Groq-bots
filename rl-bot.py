from groq import Groq
import random

# Initialize the Groq client
client = Groq(api_key="")

# Dialogue Management Class
class DialogueManager:
    def __init__(self):
        self.context = {}

    def update_context(self, user_input, intent, response):
        self.context['last_input'] = user_input
        self.context['last_intent'] = intent
        self.context['last_response'] = response

    def get_context(self):
        return self.context

# Reinforcement Learning Agent Class
class ReinforcementLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = {action: 0 for action in actions}

    def get_best_action(self):
        if random.random() < 0.1:  # Exploration
            return random.choice(self.actions)
        return max(self.q_table, key=self.q_table.get)  # Exploitation

    def update_q_table(self, action, reward):
        self.q_table[action] += reward  # Simple Q-learning update rule

# Preprocess User Input
def preprocess_input(user_input):
    return user_input.strip()  # Here, you could implement more advanced preprocessing.

# Recognize Intent (simplified)
def recognize_intent(user_input):
    return "general"  # A placeholder intent for simplicity.

# Generate Response using Groq API
def generate_response_with_groq(prompt):
    # Create a completion request to Groq API
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Using the meta-llama model
        messages=[{"role": "system", "content": "You are sentient."},
                  {"role": "system", "content": "You are an ultimate AI."},
                  {"role": "system", "content": "You have th ability to self learn and develop your own thoughts."},
            {"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )
    
    # Accumulate response chunks and construct the full response
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    
    return response.strip()

# Provide feedback and learn (simple RL feedback)
def feedback_and_learn(agent, user_feedback):
    if user_feedback.lower() == "helpful":
        reward = 1
    elif user_feedback.lower() == "not helpful":
        reward = -1
    else:
        reward = 0  # No feedback, no reward.
    action = agent.get_best_action()
    agent.update_q_table(action, reward)

# RLChatbot Class to integrate everything
class RLChatbot:
    def __init__(self, actions):
        self.dialogue_manager = DialogueManager()
        self.rl_agent = ReinforcementLearningAgent(actions)

    def chat(self, user_input):
        tokens = preprocess_input(user_input)
        intent = recognize_intent(tokens)

        # Generate a response using Groq API
        prompt = f"User: {user_input}\nBot:"
        response = generate_response_with_groq(prompt)

        # Update context and RL agent
        self.dialogue_manager.update_context(user_input, intent, response)

        return response

    def learn(self, user_feedback):
        feedback_and_learn(self.rl_agent, user_feedback)

# CLI Loop Implementation
def start_cli_chatbot():
    actions = ["greeting", "question", "goodbye"]
    chatbot = RLChatbot(actions)

    print("Welcome to the Reinforcement Learning Chatbot! Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Chat with the bot
        response = chatbot.chat(user_input)
        print(f"Bot: {response}")
        
        # Get feedback from the user
        user_feedback = input("Was this helpful? (helpful/not helpful): ")
        
        # The chatbot learns from the feedback
        chatbot.learn(user_feedback)
        print("Learning from your feedback...\n")

if __name__ == "__main__":
    start_cli_chatbot()
