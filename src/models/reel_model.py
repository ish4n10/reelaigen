import os
from typing import Dict, Any
import torch

CACHE_BASE = os.getenv("HF_CACHE_BASE", r"D:\huggingface_cache")
os.environ["HF_HOME"] = CACHE_BASE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_BASE, "models")

from transformers import pipeline   # noqa: E402
from langchain_core.output_parsers import JsonOutputParser  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402


class ReelModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_name = model_name
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=1200,
            temperature=0.7,
            # device=0 if torch.cuda.is_available() else -1,
            device=-1
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
        
        prompt = f"""You are a professional educational content creator. Analyze the following content and create a structured JSON output for educational reels.

content:{content}

sections: {sections}

CRITICAL REQUIREMENTS:
1. Follow the JSON schema exactly as specified below
2. Generate one reel entry in the "reels" array for EVERY corresponding entry in "reelsPlan"
3. Each reel's narration.estimatedSpeechSec MUST closely match its corresponding reelsPlan.targetDurationSec
4. Ensure reelId and topicId match between reelsPlan and reels entries
5. Create engaging, educational narration that fits within the target duration

{format_instructions}

VALIDATION CHECKLIST BEFORE RETURNING:
- Length of reelsPlan array equals length of reels array
- Each reel has a matching reelsPlan entry (by reelId)
- narration.estimatedSpeechSec is within ±10 seconds of targetDurationSec
- All JSON is valid and properly formatted

OUTPUT REQUIREMENTS:
- Return ONLY the JSON object
- No explanatory text before or after
- No markdown code blocks
- No comments within the JSON

Generate the JSON now:"""
        
        response = self.pipeline(prompt)[0]["generated_text"]
        generated_text = response.replace(prompt, "").strip()
        
        json_start = generated_text.find("{")
        json_end = generated_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = generated_text[json_start:json_end]
            return self.json_parser.parse(json_str)
        else:
            return self.json_parser.parse(generated_text)
