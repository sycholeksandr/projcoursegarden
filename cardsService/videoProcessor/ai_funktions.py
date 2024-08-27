import openai
from . import local_settings
from openai import OpenAI

client = OpenAI(
    api_key=local_settings.GPT_API_KEY,
)

def separate_transcription_into_themes(text):
    # Define the prompt for the model
    # Change after receiving prompt from tech dir
    prompt = f"""
    Please divide the following text into logical themes or sections.
    For each section, provide a brief title or theme and then the
    corresponding text. Format the output as follows:

    Theme: [Theme 1]
    Text: [Corresponding text]

    Theme: [Theme 2]
    Text: [Corresponding text]

    The text is: "{text}"
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.7,
    )
    completion = response.choices[0].message.content    
    return completion

def transcription_ai_cleanup(text):
    print(text)
    # Define the prompt for the model
    prompt = f"""
    Please clean the text of repetitions.
    Don't add anything by yourself.

    The text is: "{text}"
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.7,
    )
    completion = response.choices[0].message.content    
    return completion