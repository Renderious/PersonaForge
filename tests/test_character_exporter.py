import os
import json
import pytest
from character_exporter import generate_character_card_v2, generate_lorebook, export_character_data, embed_v2_in_png
from PIL import Image

def test_generate_character_card_v2():
    extracted = {
        "description": "A clever and adventurous boy.",
        "personality": ["Mischievous", "Loyal"],
        "scenario": "Living along the Mississippi River.",
        "mes_example": "{{user}}: Hi!\n{{char}}: Howdy!",
        "anachronism_guardrails": ["Internet", "Smartphones", "Airplanes"]
    }

    card = generate_character_card_v2("Tom Sawyer", "You are Tom Sawyer.", extracted)

    assert card["spec"] == "chara_card_v2"
    assert card["spec_version"] == "2.0"

    data = card["data"]
    assert data["name"] == "Tom Sawyer"
    assert data["description"] == "A clever and adventurous boy."
    assert data["personality"] == "Mischievous, Loyal"
    assert data["scenario"] == "Living along the Mississippi River."
    assert data["mes_example"] == "{{user}}: Hi!\n{{char}}: Howdy!"

    # Check guardrails injected
    assert "Internet, Smartphones, Airplanes" in data["system_prompt"]
    assert "You are Tom Sawyer." in data["system_prompt"]

def test_generate_lorebook():
    entities = [
        {"key": "Mississippi River", "content": "A very long river."},
        {"key": ["Aunt Polly", "Polly"], "content": "Tom's aunt."}
    ]

    lorebook = generate_lorebook(entities)
    entries = lorebook["entries"]

    assert len(entries) == 2
    assert entries[0]["key"] == ["Mississippi River"]
    assert entries[0]["content"] == "A very long river."
    assert entries[0]["uid"] == 1

    assert entries[1]["key"] == ["Aunt Polly", "Polly"]
    assert entries[1]["content"] == "Tom's aunt."
    assert entries[1]["uid"] == 2

def test_export_character_data(tmp_path):
    card = {"spec": "chara_card_v2"}
    lorebook = {"entries": []}

    card_path, lorebook_path = export_character_data("Tom_Sawyer", card, lorebook, str(tmp_path))

    assert os.path.exists(card_path)
    assert os.path.exists(lorebook_path)

    with open(card_path, "r") as f:
        loaded_card = json.load(f)
        assert loaded_card == card

    with open(lorebook_path, "r") as f:
        loaded_lorebook = json.load(f)
        assert loaded_lorebook == lorebook

def test_embed_v2_in_png(tmp_path):
    # Create a dummy image
    img_path = tmp_path / "dummy.png"
    out_path = tmp_path / "out.png"
    img = Image.new('RGB', (10, 10), color = 'red')
    img.save(img_path)

    card = {"test": "data"}
    embed_v2_in_png(card, str(img_path), str(out_path))

    assert os.path.exists(out_path)

    # Read the metadata to verify
    img_out = Image.open(out_path)
    assert "chara" in img_out.info

    import base64
    b64_data = img_out.info["chara"]
    json_str = base64.b64decode(b64_data).decode('utf-8')
    loaded_card = json.loads(json_str)

    assert loaded_card == card
