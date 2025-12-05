"""Multimodal metrics for cross-modal evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentunit.metrics.base import Metric, MetricResult


if TYPE_CHECKING:
    from agentunit.core.trace import TraceLog
    from agentunit.datasets.base import DatasetCase


try:
    import clip
    import torch

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from sentence_transformers import SentenceTransformer, util

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class CrossModalGroundingMetric(Metric):
    """Measures how well responses are grounded in multimodal context.

    Uses CLIP to compute similarity between visual inputs and textual responses,
    ensuring the agent's output is properly grounded in the visual context.
    """

    name = "cross_modal_grounding"

    def __init__(self, clip_model_name: str = "ViT-B/32", threshold: float = 0.25):
        """Initialize cross-modal grounding metric.

        Args:
            clip_model_name: CLIP model variant to use
            threshold: Minimum similarity threshold for grounding
        """
        self.threshold = threshold

        if not HAS_CLIP:
            msg = (
                "CLIP required for CrossModalGroundingMetric. "
                "Install with: pip install torch torchvision clip-by-openai"
            )
            raise ImportError(msg)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.device = device

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        """Evaluate cross-modal grounding."""
        image_path = case.metadata.get("image_path")
        response_text = str(outcome)

        if not image_path:
            return MetricResult(
                name=self.name, value=None, detail={"error": "No image_path in metadata"}
            )

        try:
            # Load and preprocess image
            if not HAS_PIL:
                msg = "PIL required for image processing"
                raise ImportError(msg)

            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Tokenize text
            text_tokens = clip.tokenize([response_text]).to(self.device)

            # Get features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_tokens)

            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (image_features @ text_features.T).item()

            # Score based on threshold
            score = 1.0 if similarity >= self.threshold else similarity / self.threshold

            return MetricResult(
                name=self.name,
                value=score,
                detail={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "is_grounded": similarity >= self.threshold,
                },
            )
        except Exception as e:
            return MetricResult(name=self.name, value=None, detail={"error": str(e)})


class ImageCaptionAccuracyMetric(Metric):
    """Measures accuracy of image caption generation.

    Compares generated captions against reference captions using semantic similarity
    and keyword matching.
    """

    name = "image_caption_accuracy"

    def __init__(self, use_semantic: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize image caption accuracy metric.

        Args:
            use_semantic: Whether to use semantic similarity (requires sentence-transformers)
            model_name: Sentence transformer model name
        """
        self.use_semantic = use_semantic
        self.model = None

        if use_semantic:
            if not HAS_SENTENCE_TRANSFORMERS:
                msg = (
                    "sentence-transformers required for semantic similarity. "
                    "Install with: pip install sentence-transformers"
                )
                raise ImportError(msg)
            self.model = SentenceTransformer(model_name)

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        """Evaluate caption accuracy."""
        reference_caption = case.expected_output
        generated_caption = str(outcome)

        if not reference_caption:
            return MetricResult(
                name=self.name,
                value=None,
                detail={"error": "No expected_output (reference caption) provided"},
            )

        try:
            scores = {}

            # Semantic similarity
            if self.use_semantic and self.model:
                embeddings = self.model.encode([reference_caption, generated_caption])
                semantic_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
                scores["semantic_similarity"] = semantic_sim

            # Keyword overlap (simple metric)
            ref_words = set(reference_caption.lower().split())
            gen_words = set(generated_caption.lower().split())

            if ref_words:
                precision = len(ref_words & gen_words) / len(gen_words) if gen_words else 0
                recall = len(ref_words & gen_words) / len(ref_words)
                f1 = (
                    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                )

                scores["keyword_precision"] = precision
                scores["keyword_recall"] = recall
                scores["keyword_f1"] = f1

            # Combined score (weighted average)
            if "semantic_similarity" in scores:
                final_score = 0.7 * scores["semantic_similarity"] + 0.3 * scores.get(
                    "keyword_f1", 0
                )
            else:
                final_score = scores.get("keyword_f1", 0)

            return MetricResult(name=self.name, value=final_score, detail=scores)
        except Exception as e:
            return MetricResult(name=self.name, value=None, detail={"error": str(e)})


class VideoResponseRelevanceMetric(Metric):
    """Measures relevance of responses to video content.

    For video inputs, extracts key frames and evaluates response relevance
    against visual and temporal context.
    """

    name = "video_response_relevance"

    def __init__(self, num_frames: int = 8, clip_model_name: str = "ViT-B/32"):
        """Initialize video response relevance metric.

        Args:
            num_frames: Number of frames to sample from video
            clip_model_name: CLIP model for frame analysis
        """
        self.num_frames = num_frames

        if not HAS_CLIP:
            msg = (
                "CLIP and cv2 required for VideoResponseRelevanceMetric. "
                "Install with: pip install torch torchvision clip-by-openai opencv-python"
            )
            raise ImportError(msg)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.device = device

    def _extract_frames(self, video_path: str, num_frames: int):
        """Extract evenly spaced frames from video."""
        try:
            import cv2
        except ImportError:
            msg = "opencv-python required. Install with: pip install opencv-python"
            raise ImportError(msg)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        """Evaluate video response relevance."""
        video_path = case.metadata.get("video_path")
        response_text = str(outcome)

        if not video_path:
            return MetricResult(
                name=self.name, value=None, detail={"error": "No video_path in metadata"}
            )

        try:
            if not HAS_PIL:
                msg = "PIL required for image processing"
                raise ImportError(msg)

            # Extract frames
            frames = self._extract_frames(video_path, self.num_frames)

            if not frames:
                msg = "No frames extracted from video"
                raise ValueError(msg)

            # Tokenize response
            text_tokens = clip.tokenize([response_text]).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarities = []
                for frame in frames:
                    image_input = self.preprocess(frame).unsqueeze(0).to(self.device)
                    image_features = self.model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    sim = (image_features @ text_features.T).item()
                    similarities.append(sim)

            # Aggregate similarities
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            min_similarity = min(similarities)

            # Score is weighted combination
            score = 0.6 * avg_similarity + 0.4 * max_similarity

            return MetricResult(
                name=self.name,
                value=score,
                detail={
                    "avg_frame_similarity": avg_similarity,
                    "max_frame_similarity": max_similarity,
                    "min_frame_similarity": min_similarity,
                    "num_frames_analyzed": len(similarities),
                },
            )
        except Exception as e:
            return MetricResult(name=self.name, value=None, detail={"error": str(e)})


class AudioTranscriptionMetric(Metric):
    """Measures quality of audio transcription or audio-based responses."""

    name = "audio_transcription_quality"

    def __init__(self, use_wer: bool = True):
        """Initialize audio transcription metric.

        Args:
            use_wer: Whether to compute Word Error Rate (requires jiwer)
        """
        self.use_wer = use_wer

        if use_wer:
            try:
                import jiwer

                self.jiwer = jiwer
            except ImportError:
                self.jiwer = None
                self.use_wer = False

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        """Evaluate audio transcription quality."""
        reference_text = case.expected_output
        transcribed_text = str(outcome)

        if not reference_text:
            return MetricResult(
                name=self.name,
                value=None,
                detail={"error": "No expected_output (reference transcript) provided"},
            )

        try:
            scores = {}

            # Word Error Rate
            if self.use_wer and self.jiwer:
                wer = self.jiwer.wer(reference_text, transcribed_text)
                cer = self.jiwer.cer(reference_text, transcribed_text)

                scores["wer"] = wer
                scores["cer"] = cer
                scores["accuracy"] = 1 - wer  # Convert WER to accuracy

            # Character-level similarity
            ref_chars = set(reference_text.lower())
            trans_chars = set(transcribed_text.lower())

            if ref_chars:
                char_precision = (
                    len(ref_chars & trans_chars) / len(trans_chars) if trans_chars else 0
                )
                char_recall = len(ref_chars & trans_chars) / len(ref_chars)
                scores["char_precision"] = char_precision
                scores["char_recall"] = char_recall

            # Final score
            final_score = scores.get("accuracy", scores.get("char_recall", 0))

            return MetricResult(
                name=self.name,
                value=max(0, final_score),  # Ensure non-negative
                detail=scores,
            )
        except Exception as e:
            return MetricResult(name=self.name, value=None, detail={"error": str(e)})


class MultimodalCoherenceMetric(Metric):
    """Measures coherence across multiple modalities in agent responses."""

    name = "multimodal_coherence"

    def __init__(self):
        """Initialize multimodal coherence metric."""
        self.clip_available = HAS_CLIP
        self.semantic_available = HAS_SENTENCE_TRANSFORMERS

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        """Evaluate multimodal coherence."""
        # This metric checks if the response maintains coherence when
        # referencing multiple input modalities

        input_modalities = []
        if case.metadata.get("image_path"):
            input_modalities.append("image")
        if case.metadata.get("audio_path"):
            input_modalities.append("audio")
        if case.metadata.get("video_path"):
            input_modalities.append("video")
        if case.metadata.get("text") or case.query:
            input_modalities.append("text")

        response_text = str(outcome).lower()

        # Check if response references multiple modalities
        modality_keywords = {
            "image": ["image", "picture", "photo", "visual", "see", "shown", "display"],
            "audio": ["audio", "sound", "hear", "listen", "voice", "spoken"],
            "video": ["video", "clip", "footage", "scene", "frame"],
            "text": ["text", "written", "document", "read"],
        }

        referenced_modalities = []
        for modality in input_modalities:
            keywords = modality_keywords.get(modality, [])
            if any(kw in response_text for kw in keywords):
                referenced_modalities.append(modality)

        # Score based on coverage and coherence
        if not input_modalities:
            return MetricResult(
                name=self.name, value=None, detail={"error": "No multimodal input detected"}
            )

        coverage_score = len(referenced_modalities) / len(input_modalities)

        # Check for coherence indicators (transitions, connections)
        coherence_indicators = [
            "while",
            "also",
            "additionally",
            "furthermore",
            "moreover",
            "in addition to",
            "along with",
            "together with",
            "combined with",
        ]

        coherence_count = sum(1 for indicator in coherence_indicators if indicator in response_text)
        coherence_score = min(1.0, coherence_count / 3)  # Normalize

        # Combined score
        final_score = 0.6 * coverage_score + 0.4 * coherence_score

        return MetricResult(
            name=self.name,
            value=final_score,
            detail={
                "input_modalities": input_modalities,
                "referenced_modalities": referenced_modalities,
                "coverage_score": coverage_score,
                "coherence_score": coherence_score,
                "coherence_indicators_found": coherence_count,
            },
        )


__all__ = [
    "AudioTranscriptionMetric",
    "CrossModalGroundingMetric",
    "ImageCaptionAccuracyMetric",
    "MultimodalCoherenceMetric",
    "VideoResponseRelevanceMetric",
]
