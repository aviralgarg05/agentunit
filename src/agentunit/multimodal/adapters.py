"""Multimodal adapters for vision, audio, and cross-modal processing."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class MultimodalInput:
    """Represents a multimodal input with text, image, audio, or video data."""
    
    text: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultimodalAdapter:
    """Base adapter for multimodal evaluation scenarios."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """Initialize multimodal adapter.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'gpt-4-vision-preview')
            api_key: OpenAI API key (optional, uses env var if not provided)
        """
        self.model = model
        self.api_key = api_key
        
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package required for MultimodalAdapter. "
                "Install with: pip install openai"
            )
    
    def prepare_messages(self, multimodal_input: MultimodalInput) -> List[Dict[str, Any]]:
        """Prepare messages for OpenAI API with multimodal content."""
        content = []
        
        if multimodal_input.text:
            content.append({"type": "text", "text": multimodal_input.text})
        
        if multimodal_input.image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": multimodal_input.image_url}
            })
        elif multimodal_input.image_path:
            image_data = self._encode_image(multimodal_input.image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        elif multimodal_input.image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{multimodal_input.image_base64}"}
            })
        
        return [{"role": "user", "content": content}]
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def run(self, multimodal_input: MultimodalInput, **kwargs) -> str:
        """Run inference with multimodal input."""
        import openai
        
        client = openai.AsyncOpenAI(api_key=self.api_key)
        messages = self.prepare_messages(multimodal_input)
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        return response.choices[0].message.content


class VisionAdapter(MultimodalAdapter):
    """Specialized adapter for vision-based evaluation."""
    
    def __init__(self, model: str = "gpt-4o", use_clip: bool = True, api_key: Optional[str] = None):
        """Initialize vision adapter.
        
        Args:
            model: Vision model name
            use_clip: Whether to use CLIP for embeddings
            api_key: OpenAI API key
        """
        super().__init__(model=model, api_key=api_key)
        self.use_clip = use_clip
        self.clip_model = None
        self.clip_preprocess = None
        
        if use_clip and HAS_CLIP:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.device = device
    
    def get_clip_embedding(self, image_path: str) -> Optional[torch.Tensor]:
        """Get CLIP embedding for an image."""
        if not self.clip_model or not HAS_PIL:
            return None
        
        image = Image.open(image_path)
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        
        return image_features
    
    def get_text_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Get CLIP text embedding."""
        if not self.clip_model:
            return None
        
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        
        return text_features
    
    def compute_similarity(self, image_path: str, text: str) -> Optional[float]:
        """Compute CLIP similarity between image and text."""
        if not self.clip_model:
            return None
        
        image_features = self.get_clip_embedding(image_path)
        text_features = self.get_text_embedding(text)
        
        if image_features is None or text_features is None:
            return None
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = (image_features @ text_features.T).item()
        
        return similarity


class AudioAdapter(MultimodalAdapter):
    """Specialized adapter for audio-based evaluation."""
    
    def __init__(self, model: str = "whisper-1", api_key: Optional[str] = None):
        """Initialize audio adapter.
        
        Args:
            model: Audio model name (e.g., 'whisper-1')
            api_key: OpenAI API key
        """
        super().__init__(model=model, api_key=api_key)
    
    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        import openai
        
        client = openai.AsyncOpenAI(api_key=self.api_key)
        
        with open(audio_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model=self.model,
                file=audio_file
            )
        
        return transcript.text
    
    async def run_with_audio(
        self, 
        audio_path: str, 
        prompt: str,
        vision_model: str = "gpt-4o"
    ) -> str:
        """Transcribe audio and run inference with the transcript."""
        transcript = await self.transcribe(audio_path)
        
        multimodal_input = MultimodalInput(
            text=f"{prompt}\n\nAudio Transcript: {transcript}"
        )
        
        # Use vision model for final inference
        self.model = vision_model
        return await self.run(multimodal_input)


__all__ = [
    "MultimodalInput",
    "MultimodalAdapter",
    "VisionAdapter",
    "AudioAdapter",
]
