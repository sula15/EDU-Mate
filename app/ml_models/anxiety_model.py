"""
Anxiety Detection Model Wrapper - Adapted from model.py
"""

import torch
import torch.nn as nn
import logging
from typing import Optional
from transformers import BertModel
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MultimodalFusion(nn.Module):
    """Multimodal fusion model for anxiety detection - from original model.py"""
    
    def __init__(self, num_classes=4):
        super(MultimodalFusion, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.cnn1d = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3)

        # Update LSTM input size to match CNN output
        hidden_size = 64
        self.lstm = nn.LSTM(input_size=79, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.fc_text = nn.Linear(256 * (128 // 3), 128)
        self.fc_voice = nn.Linear(2 * hidden_size, 128)  # Corrected dimension

        self.fc_final = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask, voice_input):
        # Text Path
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = self.pool(torch.relu(self.cnn1d(bert_out.permute(0, 2, 1)))).view(input_ids.size(0), -1)
        text_feat = self.fc_text(text_feat)

        # Voice Path
        lstm_out, _ = self.lstm(voice_input)  # (batch_size, seq_len, 2 * hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take the last timestep (batch_size, 2 * hidden_size)
        voice_feat = self.fc_voice(lstm_out)

        # Fusion
        fused = torch.cat((text_feat, voice_feat), dim=1)
        output = self.fc_final(fused)

        return output


# Global model instance (singleton pattern)
_anxiety_model_instance: Optional[MultimodalFusion] = None
_model_loaded = False


async def load_anxiety_model() -> MultimodalFusion:
    """Load the anxiety detection model (async for compatibility)"""
    global _anxiety_model_instance, _model_loaded
    
    if _model_loaded and _anxiety_model_instance is not None:
        return _anxiety_model_instance
    
    try:
        logger.info("Loading anxiety detection model...")
        
        # Create model instance
        model = MultimodalFusion(num_classes=4)
        
        # Load model weights if available
        model_path = settings.ANXIETY_MODEL_PATH
        if model_path and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        try:
            # Try to load the saved model
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded anxiety model weights from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model weights from {model_path}: {e}")
            logger.info("Using randomly initialized model weights")
        
        # Set model to evaluation mode
        model.eval()
        model.to(device)
        
        _anxiety_model_instance = model
        _model_loaded = True
        
        logger.info(f"Anxiety model loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading anxiety model: {e}")
        raise


def get_anxiety_model() -> Optional[MultimodalFusion]:
    """Get the loaded anxiety model (synchronous)"""
    global _anxiety_model_instance
    return _anxiety_model_instance


def is_model_loaded() -> bool:
    """Check if the anxiety model is loaded"""
    global _model_loaded
    return _model_loaded


async def unload_anxiety_model():
    """Unload the anxiety model to free memory"""
    global _anxiety_model_instance, _model_loaded
    
    if _anxiety_model_instance is not None:
        del _anxiety_model_instance
        _anxiety_model_instance = None
        _model_loaded = False
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Anxiety model unloaded")


def model_info() -> dict:
    """Get information about the anxiety model"""
    return {
        "model_name": "MultimodalFusion",
        "model_type": "BERT + LSTM fusion",
        "input_types": ["text", "audio", "multimodal"],
        "output_classes": ["No Anxiety", "Mild Anxiety", "Moderate Anxiety", "Severe Anxiety"],
        "num_classes": 4,
        "loaded": is_model_loaded(),
        "device": "cuda" if torch.cuda.is_available() and _model_loaded else "cpu"
    }