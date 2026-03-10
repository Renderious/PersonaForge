import ollama
import json

# Default model to use. 'llama3' or 'mistral' are good choices if installed locally via Ollama.
# The user might need to adjust this depending on what they have downloaded (`ollama run llama3`)
DEFAULT_MODEL = "llama3"

def extract_characters(text_chunks: list[str], model: str = DEFAULT_MODEL) -> list[str]:
    """
    Passes a sample of text chunks to the LLM to identify the primary characters.

    Args:
        text_chunks (list[str]): The chunks of the book.
        model (str): The Ollama model to use.

    Returns:
        list[str]: A list of primary character names extracted from the text.
    """
    # For extraction, we don't need the entire book.
    # Let's take a sample of chunks (e.g., the first few and some from the middle) to find main characters.
    sample_size = min(10, len(text_chunks))
    sample_chunks = text_chunks[:sample_size]

    combined_text = "\n\n".join(sample_chunks)

    prompt = (
        "Analyze the following text from a book and extract the names of the primary characters. "
        "Return a JSON object with a single key 'characters' containing a list of strings. "
        "For example: {\"characters\": [\"Alice\", \"Bob\", \"Charlie\"]}\n\n"
        f"Text:\n{combined_text}"
    )

    try:
        # Using format='json' forces Ollama to output valid JSON
        response = ollama.chat(model=model, format='json', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])

        content = response['message']['content']

        # Try to parse the JSON response. The LLM might include markdown formatting.
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()

        # Parse the JSON response
        data = json.loads(content)

        # Extract the list from the 'characters' key
        if 'characters' in data and isinstance(data['characters'], list):
            return data['characters']
        else:
            print("Warning: LLM did not return the expected JSON structure.")
            return []

    except Exception as e:
        print(f"Error during character extraction: {e}")
        return []


def generate_character_system_prompt(character_name: str, text_chunks: list[str], model: str = DEFAULT_MODEL) -> str:
    """
    Generates a concise Core System Prompt focusing on the character's tone, personality, and speaking style.

    Args:
        character_name (str): The name of the character.
        text_chunks (list[str]): Chunks containing the character (ideally filtered for relevance).
        model (str): The Ollama model to use.

    Returns:
        str: The generated system prompt.
    """
    # Combine chunks to provide context for the character's persona
    combined_text = "\n\n".join(text_chunks)

    prompt = (
        f"Based on the following text excerpts from a book, analyze the character '{character_name}'. "
        "Create a concise 'Core System Prompt' that describes their tone, personality, and speaking style. "
        "Do NOT include their entire life history or plot details. Focus ONLY on how they speak and act. "
        "Write it as a set of instructions for an AI adopting this persona (e.g., 'You are [Name]. You speak with...').\n\n"
        f"Text Excerpts:\n{combined_text}"
    )

    try:
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error generating system prompt for {character_name}: {e}")
        return f"You are {character_name}. Embody this character based on the user's input."


def chat_stream(character_name: str, system_prompt: str, context_chunks: list[str], user_message: str, model: str = DEFAULT_MODEL):
    """
    Handles the chat loop, combining system prompt, RAG context, and the user message.
    Streams the response back.

    Args:
        character_name (str): The name of the character being roleplayed.
        system_prompt (str): The core system prompt for the character.
        context_chunks (list[str]): Relevant chunks retrieved from the vector DB.
        user_message (str): The user's input message.
        model (str): The Ollama model to use.

    Yields:
        str: Chunks of the LLM's response.
    """

    # Combine the context chunks into a single string for memory
    memory_context = "\n\n".join(context_chunks)

    # Build the full system message
    full_system_message = (
        f"{system_prompt}\n\n"
        "Here are some relevant memories/excerpts from the book to help you respond accurately:\n"
        f"{memory_context}\n\n"
        "Respond to the user in character."
    )

    try:
        stream = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': full_system_message},
                {'role': 'user', 'content': user_message}
            ],
            stream=True
        )

        for chunk in stream:
            yield chunk['message']['content']

    except Exception as e:
        yield f"\n\n[Error communicating with Ollama: {e}]"


def extract_character_details(character_name: str, text_chunks: list[str], model: str = DEFAULT_MODEL) -> dict:
    """
    Passes text chunks to the LLM to extract character traits, dialect, relationships, and anachronism guardrails.
    Returns a dictionary of these details.
    """
    combined_text = "\n\n".join(text_chunks)

    prompt = (
        f"Analyze the following text from a book and extract detailed information about the character '{character_name}'.\n"
        "Return a JSON object with the following keys:\n"
        "- 'description': Physical appearance and general background.\n"
        "- 'personality': Character traits and psychological profile.\n"
        "- 'scenario': The general setting or current situation they are in.\n"
        "- 'mes_example': Example dialogue lines showing their dialect and speaking style (format as \"{{user}}: ...\n{{char}}: ...\").\n"
        "- 'anachronism_guardrails': A list of things this character should NOT know based on their time period or setting.\n\n"
        f"Text:\n{combined_text}"
    )

    try:
        response = ollama.chat(model=model, format='json', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])

        content = response['message']['content']

        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()

        data = json.loads(content)
        return data

    except Exception as e:
        print(f"Error during character details extraction: {e}")
        return {}


def extract_world_entities(text_chunks: list[str], model: str = DEFAULT_MODEL, progress_callback=None) -> list[dict]:
    """
    Passes text chunks to the LLM to extract world-building entities for a Lorebook.
    Processes chunks in batches to avoid context limits.
    Returns a list of dictionaries with 'key' and 'content'.
    """
    all_entities = []

    # Process in batches of 5 chunks
    batch_size = 5
    total_batches = (len(text_chunks) + batch_size - 1) // batch_size
    for i in range(0, len(text_chunks), batch_size):
        current_batch = i // batch_size
        if progress_callback:
            progress_callback(current_batch, total_batches)

        batch = text_chunks[i:i + batch_size]
        combined_text = "\n\n".join(batch)

        prompt = (
            "Analyze the following text from a book and extract key world-building entities (locations, items, concepts, secondary characters).\n"
            "Return a JSON object with a single key 'entities' containing a list of objects. Each object should have a 'key' (the entity name) and 'content' (a brief description).\n"
            "For example: {\"entities\": [{\"key\": \"The Ring\", \"content\": \"A magical artifact...\"}]}\n\n"
            f"Text:\n{combined_text}"
        )

        try:
            response = ollama.chat(model=model, format='json', messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])

            content = response['message']['content']

            # Robust JSON parsing
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            # Find JSON object boundaries if extra text is present
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]

            data = json.loads(content)

            if 'entities' in data and isinstance(data['entities'], list):
                all_entities.extend(data['entities'])

        except Exception as e:
            print(f"Error during world entity extraction batch {i//batch_size}: {e}")
            continue

    # Deduplicate entities based on key
    unique_entities = {}
    for entity in all_entities:
        key = entity.get("key")
        if key and key not in unique_entities:
            unique_entities[key] = entity

    return list(unique_entities.values())
