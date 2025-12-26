from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from pydub import AudioSegment
import os
import tempfile
import shutil
import re
from typing import List, Dict, Optional

class VoiceModel:
    def __init__(self, model_name: str = "microsoft/speecht5_tts", 
                 vocoder_name: str = "microsoft/speecht5_hifigan"):
        
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.vocoder.to(self.device)
        
        self.speaker_embeddings = self._generate_stable_embedding()

    def _generate_stable_embedding(self):
        """Creates a consistent vocal profile to prevent 'grainy' audio."""
        torch.manual_seed(42) 
        embedding = torch.randn(1, 512) * 0.05
        return embedding.to(self.device)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Prevents stuttering by ensuring text segments aren't too long."""
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s.strip()]

    def _change_speed(self, audio: AudioSegment, speed: float) -> AudioSegment:
        """Adjusts the playback speed (1.2 is 20% faster, 0.8 is 20% slower)."""
        if speed == 1.0:
            return audio
        new_sample_rate = int(audio.frame_rate * speed)
        return audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(audio.frame_rate)

    def generate_audio_from_segments(self, segments: List[Dict], output_path: str = "final_output.mp3", speed: float = 1.0):
        """Generates clear speech with speed control."""
        temp_dir = tempfile.mkdtemp()
        audio_files = []

        try:
            for i, seg in enumerate(segments):
                sub_sentences = self._split_into_sentences(seg["text"])
                
                segment_audio = AudioSegment.silent(duration=0)
                for j, sentence in enumerate(sub_sentences):
                    tmp_wav = os.path.join(temp_dir, f"seg_{i}_{j}.wav")
                    
                    inputs = self.processor(text=sentence, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
                    
                    sf.write(tmp_wav, speech.cpu().numpy(), samplerate=16000)
                    
                    sentence_audio = AudioSegment.from_wav(tmp_wav)
                    segment_audio += sentence_audio

                segment_audio = self._change_speed(segment_audio, speed)
                
                final_wav = os.path.join(temp_dir, f"final_seg_{i}.wav")
                segment_audio.export(final_wav, format="wav")
                audio_files.append((final_wav, seg["start"], seg["end"]))

            final_combined = self._stitch_audio_segments(audio_files)
            final_combined.export(output_path, format="mp3")
            print(f"Success! Saved to {output_path}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _stitch_audio_segments(self, audio_files: List[tuple]) -> AudioSegment:
        """Combines segments with correct timing."""
        final_audio = AudioSegment.silent(duration=0)
        for filename, start, end in audio_files:
            segment_audio = AudioSegment.from_wav(filename)
            silence_padding = (start * 1000) - len(final_audio)
            if silence_padding > 0:
                final_audio += AudioSegment.silent(duration=silence_padding)
            final_audio += segment_audio
        return final_audio

if __name__ == "__main__":
    model = VoiceModel()
    
    my_segments = [
        {
            "text": "Understanding the architecture of the Android platform is crucial for improving its security. We'll examine the various components and identify potential security weaknesses.", 
            "start": 0, "end": 8
        },
        {
            "text": "Now we continue after a pause.", 
            "start": 10, "end": 13
        }
    ]
    
    model.generate_audio_from_segments(my_segments, speed=1.1)