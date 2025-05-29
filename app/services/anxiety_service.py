"""
Anxiety Detection Service - Adapted from anxiety_detection.py
"""

import os
import tempfile
import logging
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import torch
import librosa
import numpy as np
import whisper
from transformers import BertTokenizer

from app.models.anxiety import (
    AnxietyAssessmentRequest, AnxietyAssessment, AnxietyPrediction,
    WellnessRecommendation, AnxietyLevel, InputType, WellnessRecommendationType,
    AnxietyFeedback, AnxietyAnalytics, AnxietyTrend
)
from app.core.config import get_settings
from app.core.database import get_mongo_collection

logger = logging.getLogger(__name__)
settings = get_settings()


class AnxietyService:
    """Service for anxiety detection and wellness recommendations"""
    
    def __init__(self):
        self.tokenizer: Optional[BertTokenizer] = None
        self.anxiety_model = None
        self.whisper_model = None
        self._models_loaded = False
        
        # Anxiety level mappings
        self.anxiety_levels = [
            AnxietyLevel.NO_ANXIETY,
            AnxietyLevel.MILD_ANXIETY, 
            AnxietyLevel.MODERATE_ANXIETY,
            AnxietyLevel.SEVERE_ANXIETY
        ]
        
        # Response messages for each level
        self.response_messages = {
            AnxietyLevel.NO_ANXIETY: "You seem to be doing well. Keep up the good work!",
            AnxietyLevel.MILD_ANXIETY: "It seems like you are having mild anxiety. Consider talking to a counselor or engaging in relaxation techniques.",
            AnxietyLevel.MODERATE_ANXIETY: "It seems like you are experiencing moderate anxiety. It might be helpful to talk to a mental health professional.",
            AnxietyLevel.SEVERE_ANXIETY: "It seems like you are experiencing severe anxiety. It's important to seek help from a professional as soon as possible."
        }
    
    async def load_models(self) -> bool:
        """Load anxiety detection and speech recognition models"""
        if self._models_loaded:
            return True
            
        try:
            logger.info("Loading anxiety detection models...")
            
            # Load BERT tokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            
            # Load anxiety detection model
            from app.ml_models.anxiety_model import load_anxiety_model
            self.anxiety_model = await load_anxiety_model()
            
            # Load Whisper model for speech recognition
            self.whisper_model = whisper.load_model(settings.WHISPER_MODEL_SIZE)
            
            self._models_loaded = True
            logger.info("Successfully loaded all anxiety detection models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading anxiety models: {e}")
            return False
    
    def tokenize_and_pad(self, text: str, max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and pad text input for the model"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
            
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']
    
    def preprocess_audio(self, audio_path: str, sample_rate: int = 16000, 
                        n_mels: int = 64, duration: float = 2.5) -> torch.Tensor:
        """Process audio file for the anxiety detection model"""
        try:
            # Load audio
            waveform, sr = librosa.load(audio_path, sr=None)
            
            # Resample if necessary
            if sr != sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
            
            # Ensure consistent length
            num_samples = int(sample_rate * duration)
            if len(waveform) < num_samples:
                waveform = np.pad(waveform, (0, num_samples - len(waveform)))
            else:
                waveform = waveform[:num_samples]
            
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return torch.tensor(mel_spec_db).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            # Return zero tensor as fallback
            return torch.zeros((1, n_mels, 79))
    
    async def transcribe_audio(self, audio_data: bytes, audio_format: str = "wav") -> str:
        """Transcribe audio data to text using Whisper"""
        if not self.whisper_model:
            raise ValueError("Whisper model not loaded")
        
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}') as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(temp_audio_path)
            transcript = result["text"].strip()
            
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    async def predict_anxiety(self, text: Optional[str] = None, 
                            audio_tensor: Optional[torch.Tensor] = None) -> AnxietyPrediction:
        """Predict anxiety level from text and/or audio input"""
        if not self.anxiety_model:
            raise ValueError("Anxiety model not loaded")
        
        start_time = datetime.now()
        
        try:
            # Prepare inputs
            if text:
                input_ids, attention_mask = self.tokenize_and_pad(text)
                if audio_tensor is None:
                    audio_tensor = torch.zeros((1, 64, 79))
            else:
                input_ids = torch.zeros((1, 128), dtype=torch.long)
                attention_mask = torch.zeros((1, 128), dtype=torch.long)
                if audio_tensor is None:
                    audio_tensor = torch.zeros((1, 64, 79))
            
            # Determine input type
            input_type = InputType.TEXT_ONLY if text and audio_tensor.sum() == 0 else \
                        InputType.AUDIO_ONLY if not text and audio_tensor.sum() > 0 else \
                        InputType.MULTIMODAL
            
            # Make prediction
            with torch.no_grad():
                output = self.anxiety_model(input_ids, attention_mask, audio_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_label = torch.argmax(output).item()
                confidence_score = probabilities[0][predicted_label].item()
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnxietyPrediction(
                anxiety_level=self.anxiety_levels[predicted_label],
                confidence_score=confidence_score,
                input_type=input_type,
                processing_time=processing_time,
                model_version="1.0"
            )
            
        except Exception as e:
            logger.error(f"Error predicting anxiety: {e}")
            # Return default prediction
            processing_time = (datetime.now() - start_time).total_seconds()
            return AnxietyPrediction(
                anxiety_level=AnxietyLevel.MILD_ANXIETY,
                confidence_score=0.0,
                input_type=InputType.TEXT_ONLY,
                processing_time=processing_time,
                model_version="1.0"
            )
    
    def get_wellness_recommendations(self, anxiety_level: AnxietyLevel) -> List[WellnessRecommendation]:
        """Get personalized wellness recommendations based on anxiety level"""
        
        if anxiety_level == AnxietyLevel.NO_ANXIETY:
            return [
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.MINDFULNESS,
                    title="Maintain Your Well-being",
                    description="Keep up your great mental health with these maintenance practices",
                    instructions=[
                        "Continue your current healthy habits",
                        "Practice gratitude daily",
                        "Maintain regular sleep schedule",
                        "Stay connected with friends and family"
                    ],
                    duration_minutes=10,
                    difficulty_level=1
                )
            ]
        
        elif anxiety_level == AnxietyLevel.MILD_ANXIETY:
            return [
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.BREATHING,
                    title="Deep Breathing Exercise",
                    description="Simple breathing techniques to reduce mild anxiety",
                    instructions=[
                        "Find a comfortable seated position",
                        "Breathe in slowly through your nose for 4 counts",
                        "Hold your breath for 4 counts",
                        "Exhale slowly through your mouth for 6 counts",
                        "Repeat 5-10 times"
                    ],
                    duration_minutes=5,
                    difficulty_level=1
                ),
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.PHYSICAL,
                    title="Short Walk Outside",
                    description="Light physical activity to reduce anxiety",
                    instructions=[
                        "Step outside for fresh air",
                        "Walk at a comfortable pace",
                        "Focus on your surroundings",
                        "Take deep breaths while walking",
                        "Aim for 10-15 minutes"
                    ],
                    duration_minutes=15,
                    difficulty_level=1
                ),
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.MINDFULNESS,
                    title="5-Minute Mindfulness",
                    description="Quick mindfulness meditation for anxiety relief",
                    instructions=[
                        "Sit quietly and close your eyes",
                        "Focus on your breathing",
                        "Notice thoughts without judgment",
                        "Gently return attention to breath",
                        "Practice for 5 minutes"
                    ],
                    duration_minutes=5,
                    difficulty_level=2
                )
            ]
        
        elif anxiety_level == AnxietyLevel.MODERATE_ANXIETY:
            return [
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.BREATHING,
                    title="5-4-3-2-1 Grounding Technique",
                    description="Grounding technique to manage moderate anxiety",
                    instructions=[
                        "Name 5 things you can see",
                        "Name 4 things you can touch",
                        "Name 3 things you can hear",
                        "Name 2 things you can smell",
                        "Name 1 thing you can taste",
                        "Take deep breaths between each step"
                    ],
                    duration_minutes=10,
                    difficulty_level=2
                ),
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.PHYSICAL,
                    title="Progressive Muscle Relaxation",
                    description="Systematic muscle relaxation to reduce tension",
                    instructions=[
                        "Start with your toes and work upward",
                        "Tense each muscle group for 5 seconds",
                        "Release and relax for 10 seconds",
                        "Notice the contrast between tension and relaxation",
                        "Continue through all muscle groups"
                    ],
                    duration_minutes=20,
                    difficulty_level=3
                ),
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.SOCIAL,
                    title="Reach Out to Someone",
                    description="Connect with trusted friends or family",
                    instructions=[
                        "Identify a trusted friend or family member",
                        "Call or message them about how you're feeling",
                        "Share your current concerns",
                        "Ask for their support or just to listen",
                        "Consider scheduling regular check-ins"
                    ],
                    duration_minutes=30,
                    difficulty_level=2
                )
            ]
        
        else:  # SEVERE_ANXIETY
            return [
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.PROFESSIONAL,
                    title="Seek Professional Help",
                    description="Important resources for severe anxiety",
                    instructions=[
                        "Contact your university's counseling services immediately",
                        "Call your healthcare provider",
                        "Consider calling a crisis helpline if needed",
                        "Reach out to a trusted adult or friend",
                        "Don't face this alone - help is available"
                    ],
                    duration_minutes=0,
                    difficulty_level=1
                ),
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.BREATHING,
                    title="Emergency Breathing Technique",
                    description="Immediate anxiety relief breathing exercise",
                    instructions=[
                        "Sit or lie down in a safe space",
                        "Breathe in for 4 counts through your nose",
                        "Hold for 7 counts",
                        "Exhale completely for 8 counts through your mouth",
                        "Repeat until you feel calmer",
                        "Remember: this feeling will pass"
                    ],
                    duration_minutes=10,
                    difficulty_level=1
                ),
                WellnessRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    type=WellnessRecommendationType.SOCIAL,
                    title="Emergency Contacts",
                    description="Immediate support resources",
                    instructions=[
                        "Crisis Text Line: Text HOME to 741741",
                        "National Suicide Prevention Lifeline: 988",
                        "University Counseling Center: Check your campus directory",
                        "Campus Security: Available 24/7",
                        "Trusted emergency contact: Call immediately"
                    ],
                    duration_minutes=0,
                    difficulty_level=1
                )
            ]
    
    async def assess_anxiety(self, request: AnxietyAssessmentRequest) -> AnxietyAssessment:
        """Perform complete anxiety assessment"""
        # Ensure models are loaded
        if not await self.load_models():
            raise RuntimeError("Failed to load anxiety detection models")
        
        assessment_id = str(uuid.uuid4())
        transcript = None
        audio_tensor = None
        
        try:
            # Process audio if provided
            if request.audio_data:
                # Decode base64 audio data
                import base64
                audio_bytes = base64.b64decode(request.audio_data)
                
                # Transcribe audio
                transcript = await self.transcribe_audio(
                    audio_bytes, 
                    request.audio_format or "wav"
                )
                
                # Process audio for anxiety detection
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_file.write(audio_bytes)
                    temp_audio_path = temp_file.name
                
                audio_tensor = self.preprocess_audio(temp_audio_path)
                
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
            
            # Use text input or transcript
            text_input = request.input_text or transcript
            
            # Predict anxiety level
            prediction = await self.predict_anxiety(text_input, audio_tensor)
            
            # Get recommendations
            recommendations = self.get_wellness_recommendations(prediction.anxiety_level)
            
            # Create assessment
            assessment = AnxietyAssessment(
                assessment_id=assessment_id,
                student_id=request.student_id,
                prediction=prediction,
                recommendations=recommendations,
                input_text=text_input,
                has_audio=request.audio_data is not None,
                transcript=transcript,
                timestamp=datetime.now()
            )
            
            # Store assessment in database
            await self.save_assessment(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in anxiety assessment: {e}")
            raise
    
    async def save_assessment(self, assessment: AnxietyAssessment) -> bool:
        """Save anxiety assessment to database"""
        try:
            # Save to anxiety database
            anxiety_collection = get_mongo_collection("anxiety_assessments")
            assessment_data = assessment.model_dump()
            
            # Convert datetime to ISO string for MongoDB
            assessment_data["timestamp"] = assessment.timestamp.isoformat()
            
            anxiety_collection.insert_one(assessment_data)
            
            # Also save to student profile if available
            try:
                profiles_collection = get_mongo_collection(settings.STUDENT_PROFILES_COLLECTION)
                profiles_collection.update_one(
                    {"_id": assessment.student_id},
                    {"$push": {"anxiety_assessments": {
                        "assessment_id": assessment.assessment_id,
                        "timestamp": assessment.timestamp,
                        "anxiety_level": assessment.prediction.anxiety_level,
                        "confidence_score": assessment.prediction.confidence_score,
                        "has_audio": assessment.has_audio
                    }}}
                )
            except Exception as profile_err:
                logger.warning(f"Could not update student profile: {profile_err}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving anxiety assessment: {e}")
            return False
    
    async def get_student_anxiety_history(self, student_id: str, days: int = 30) -> List[AnxietyAssessment]:
        """Get anxiety assessment history for a student"""
        try:
            anxiety_collection = get_mongo_collection("anxiety_assessments")
            
            # Calculate date threshold
            from datetime import timedelta
            threshold_date = datetime.now() - timedelta(days=days)
            
            # Query assessments
            cursor = anxiety_collection.find({
                "student_id": student_id,
                "timestamp": {"$gte": threshold_date.isoformat()}
            }).sort("timestamp", -1)
            
            assessments = []
            for doc in cursor:
                try:
                    # Convert back to AnxietyAssessment model
                    doc["timestamp"] = datetime.fromisoformat(doc["timestamp"])
                    assessment = AnxietyAssessment.model_validate(doc)
                    assessments.append(assessment)
                except Exception as e:
                    logger.error(f"Error parsing assessment: {e}")
                    continue
            
            return assessments
            
        except Exception as e:
            logger.error(f"Error getting anxiety history: {e}")
            return []
    
    async def submit_feedback(self, feedback: AnxietyFeedback) -> bool:
        """Submit feedback for an anxiety assessment"""
        try:
            feedback_collection = get_mongo_collection("anxiety_feedback")
            feedback_data = feedback.model_dump()
            feedback_data["timestamp"] = feedback.timestamp.isoformat()
            
            feedback_collection.insert_one(feedback_data)
            return True
            
        except Exception as e:
            logger.error(f"Error submitting anxiety feedback: {e}")
            return False
    
    async def get_anxiety_analytics(self, student_id: str) -> Optional[AnxietyAnalytics]:
        """Generate anxiety analytics for a student"""
        try:
            # Get assessment history
            assessments = await self.get_student_anxiety_history(student_id, days=90)
            
            if not assessments:
                return None
            
            # Calculate analytics
            total_assessments = len(assessments)
            
            # Level distribution
            level_counts = {}
            for level in AnxietyLevel:
                level_counts[level] = sum(1 for a in assessments if a.prediction.anxiety_level == level)
            
            # Percentages
            level_percentages = {
                level: (count / total_assessments) * 100 
                for level, count in level_counts.items()
            }
            
            # Trend data (simplified)
            trend_data = []
            for assessment in assessments[-30:]:  # Last 30 assessments
                trend_data.append(AnxietyTrend(
                    student_id=student_id,
                    date=assessment.timestamp,
                    anxiety_level=assessment.prediction.anxiety_level,
                    confidence_score=assessment.prediction.confidence_score,
                    assessment_count=1,
                    avg_confidence=assessment.prediction.confidence_score
                ))
            
            # Determine overall trend (simplified)
            if len(assessments) >= 2:
                recent_avg = np.mean([
                    list(AnxietyLevel).index(a.prediction.anxiety_level) 
                    for a in assessments[:5]
                ])
                older_avg = np.mean([
                    list(AnxietyLevel).index(a.prediction.anxiety_level) 
                    for a in assessments[-5:]
                ])
                
                if recent_avg < older_avg:
                    overall_trend = "improving"
                elif recent_avg > older_avg:
                    overall_trend = "declining"
                else:
                    overall_trend = "stable"
            else:
                overall_trend = "stable"
            
            # Risk assessment
            severe_count = level_counts.get(AnxietyLevel.SEVERE_ANXIETY, 0)
            moderate_count = level_counts.get(AnxietyLevel.MODERATE_ANXIETY, 0)
            
            if severe_count > 0 or (moderate_count / total_assessments) > 0.5:
                risk_level = "high"
                intervention_recommended = True
            elif (moderate_count / total_assessments) > 0.3:
                risk_level = "medium"
                intervention_recommended = False
            else:
                risk_level = "low"
                intervention_recommended = False
            
            return AnxietyAnalytics(
                student_id=student_id,
                total_assessments=total_assessments,
                assessment_period_days=90,
                level_distribution=level_counts,
                level_percentages=level_percentages,
                trend_data=trend_data,
                overall_trend=overall_trend,
                risk_level=risk_level,
                intervention_recommended=intervention_recommended,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating anxiety analytics: {e}")
            return None