import os
from typing import Dict, Any

CACHE_BASE = os.getenv("HF_CACHE_BASE", r"G:\huggingface_cache")
os.environ["HF_HOME"] = CACHE_BASE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_BASE, "models")

from transformers import pipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class ReelModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_name = model_name
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=500,
            temperature=0.7
        )
        self.json_parser = JsonOutputParser()
    
    def generate_reel_data(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        content = content_data["allText"][:1500]
        sections = content_data["noOfSep"]
        
        format_instructions = """Return JSON in this exact format:

{
  "documentMeta": {
    "title": "string",
    "audienceType": "string",
    "difficulty": "string"
  },
  "topics": [
    {
      "id": "string",
      "name": "string",
      "relevance": number,
      "estimatedDepth": "string"
    }
  ],
  "reelsPlan": [
    {
      "reelId": "string",
      "topicId": "string",
      "title": "string",
      "targetDurationSec": number,
      "audienceLevel": "string",
      "keyPoints": ["string"]
    }
  ],
  "reels": [
    {
      "reelId": "string",
      "topicId": "string",
      "narration": {
        "text": "string",
        "estimatedSpeechSec": number
      },
      "explanationLevel": "string",
      "visualIntent": [
        {
          "type": "string",
          "content": "string"
        }
      ]
    }
  ]
}"""
        
        prompt = f"""Create JSON for educational reels from this content:

Content: {content}
Sections: {sections}

{format_instructions}

Return ONLY valid JSON, no other text:"""
        
        response = self.pipeline(prompt)[0]["generated_text"]
        generated_text = response.replace(prompt, "").strip()
        
        json_start = generated_text.find("{")
        json_end = generated_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = generated_text[json_start:json_end]
            return self.json_parser.parse(json_str)
        else:
            return self.json_parser.parse(generated_text)
