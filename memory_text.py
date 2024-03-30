# ChatMemory class that manages the chat history in a single channel

import pickle
from mistralai.models.chat_completion import ChatMessage

AI_NAME = "Assistant"


class ChatMemory:
    def __init__(self, max_message_limit=20, name=AI_NAME, prompt=None):
        """
        Set defaults for chat memory

        :param max_message_limit: Maximum number of messages to store in memory
        :param name: Name of the AI
        :param prompt: Prompt to use for the AI
        """

        print("Initializing chat memory")
        print("Max message limit: " + str(max_message_limit))
        self.memory = []
        self.max_message_limit = max_message_limit
        self.name = AI_NAME
        self.prompt = prompt

    def add(self, chat):
        """
        Add a message(s) to the chat memory

        :param name: Name of the user that sent the message
        :param *message: Message(s) to add to the chat memory
        """
        self.memory.append(chat)
        self.clean()
        return self

    def construct_history(self, list=True):
        """
        Construct the chat history from the chat memory

        :param list: Whether to return the history as a list or a string
        """

        history = []
        for message in self.memory:
            history.append(message)
        if not list:
            return "\n".join(history)
        return history
    
    def construct_history_print(self, list=True):
        """
        Construct the chat history from the chat memory

        :param list: Whether to return the history as a list or a string
        """

        history = []
        for message in self.memory:
            history.append(f"{message.role}: {message.content}")
        if not list:
            return "\n".join(history)
        return history

    def clean(self):
        """
        Clean the chat memory to ensure it does not exceed the maximum message limit
        Runs after every message is added to the chat memory
        """
        # Get count of messages in memory
        # While the total number of tokens is above the limit
        while len(self.memory) > self.max_message_limit:
            # Remove the oldest message from the memory
            oldest_message = self.memory.pop(0)
            print(f"Removing message from chat memory: {oldest_message}")

        return self


    def clear(self):
        """
        Clear the chat memory
        """
        self.memory.clear()
        if self.prompt is not None:
            self.memory.append(self.prompt)

    def save(self):
        """
        Save the chat memory to memory.pickle
        """
        with open("memory.pickle", "wb") as f:
            pickle.dump(self.memory, f)

        return self

    def load(self):
        """
        Load the chat memory from memory.pickle
        """
        with open("memory.pickle", "rb") as f:
            self.memory = pickle.load(f)

        return self
    
    def remove_idx(self, idx):
        """
        Remove a message from the chat memory by index

        :param idx: Index of the message to remove
        """
        self.memory.pop(idx)
        return self

    def set_prompt(self, prompt):
        """
        Set the prompt for the AI
        Will be inserted at the beginning of the chat memory

        :param prompt: Prompt to use for the AI
        """
        self.prompt = ChatMessage(role="system", content=prompt)
        self.memory.insert(0, self.prompt)
