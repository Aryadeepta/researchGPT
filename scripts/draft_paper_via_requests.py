import os
import json
import requests
import argparse

# Direct API call to bypass google-genai library if it hangs
def draft_section_via_requests(section_name, topic, research_context):
    api_key = os.environ.get("GOOGLE_API_KEY")
    model = "gemini-3.1-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    prompt = f"""
Draft the content for the {section_name} section of a paper titled '{topic}'.
Ensure it is professional, academic, and detailed. Use LaTeX for math.
Context:
{research_context}

Return ONLY the LaTeX content for this section.
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        raise Exception(f"Requests API failed with status {response.status_code}: {response.text}")

if __name__ == "__main__":
    # This script is a simplified alternative to draft_paper
    print("Standalone requester started.")
    # You would need to load state and call this for each section...
    # This is provided as a debugging/fallback utility.
    pass
