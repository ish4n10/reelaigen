import os
from typing import Dict, Any

CACHE_BASE = os.getenv("HF_CACHE_BASE", r"G:\huggingface_cache")
os.environ["HF_HOME"] = CACHE_BASE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_BASE, "models")

from transformers import pipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from manim_helpers._templates import template_map


class ManimModel: 
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        # Extract template names from template_map keys (remove function signature part)
        self.templates = [f"{key.split('(')[0]} {{}}" for key in template_map.keys()]
        self.model_name = model_name
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=500,
            temperature=0.7,
            device=-1
        )
        self.json_parser = JsonOutputParser()


    def generate_manim_data(self: 'ManimModel', content_text: str, time_length: int) -> Dict[str, Any]:
        
        content = content_text[:1500]
        
        parsed_template_array = ', '.join(self.templates)


        format_instructions = f"""Return JSON in this exact format:

{{
  "manimTemplateList": [
    {{
      "startTime": 0,
      "endTime": 5,
      "templateName": "text_intro {{}}",
      "content": "string content to pass to the template function",
      "narration": {{
        "speechTone": "string describing the tone (e.g., 'explanatory', 'enthusiastic', 'calm', 'professional')",
        "text": "string with the narration text that explains what is happening in the scene"
      }},
      "timestamp": 0.0,
      "sceneDescription": "string describing the scene",
      "explanationLevel": "string describing the explanation level"
    }}
  ]
}}

Important:
- startTime and endTime must be numbers (not strings)
- templateName must be one of the available templates: {parsed_template_array}
- content must be a string that will be passed as the first argument to the template function
- narration must be an object containing:
  - speechTone: a string describing the speech/audio tone appropriate for an explanatory video (e.g., "explanatory", "enthusiastic", "calm", "professional")
  - text: a string containing the narration text that explains what is happening in the current scene, suitable for audio/speech in an explanatory video
- timestamp must be a number (float) that will be passed as the second argument to the template function (typically use startTime or endTime)
- sceneDescription should describe what will be shown in the manim scene
- explanationLevel should indicate the level of detail (e.g., "basic", "intermediate", "advanced")
- Create multiple entries in manimTemplateList to cover the entire video duration
- Ensure endTime of one entry matches startTime of the next entry
- Total duration should match the time_length provided"""

        prompt_template = """You are an expert content analyzer for creating Manim educational videos. 

Analyze the following content and create a structured timeline for a Manim video.

Content to analyze:
{content}

Total video duration: {time_length} seconds

{format_instructions}

Based on the content provided, break it down into scenes with appropriate timing. Return ONLY valid JSON in the exact format specified above. Do not include any additional text before or after the JSON."""

        formatted_prompt = prompt_template.format(
            content=content,
            time_length=time_length,
            format_instructions=format_instructions
        )
        
        response = self.pipeline(formatted_prompt)[0]["generated_text"]
        generated_text = response.replace(formatted_prompt, "").strip()

        json_start = generated_text.find("{")
        json_end = generated_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = generated_text[json_start:json_end]
            return self.json_parser.parse(json_str)
        else:
            return self.json_parser.parse(generated_text)
        