import openai

# Set your API key
openai.api_key =  "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"

# Initialize the conversation with a system message
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
]

while True:
    # Get a message from the user
    user_message = input("You: ")

    # Add the user message to the conversation
    conversation.append({"role": "user", "content": user_message})

    # Get a response from GPT-3
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Extract the assistant's message
    assistant_message = response['choices'][0]['message']['content']

    # Print the assistant's message
    print("Assistant: ", assistant_message)

    # Add the assistant's message to the conversation
    conversation.append({"role": "assistant", "content": assistant_message})
