import pandas as pd
import requests
import time
from datetime import datetime
from typing import Dict

class DualValidationSystem:
    """Two-stage validation: ML model + LLM cross-check with asymmetric flip rules"""
    
    def __init__(self, trained_model, confidence_threshold=0.3):
        self.ml_model = trained_model
        self.confidence_threshold = confidence_threshold
        self.session = requests.Session()
        self.base_url = "http://localhost:11434/api/generate"
        self.model_name = "llama2"
        
        self.stats = {
            'total_predictions': 0,
            'cross_checks_performed': 0,
            'agreements': 0,
            'disagreements': 0,
            'flips_to_fake': 0,
            'flips_to_legit': 0
        }
    
    def create_cross_check_prompt(self, review_data: Dict, ml_prediction: Dict) -> str:
        """Create few-shot prompt for LLM cross-validation"""
        
        prompt = f"""You are validating an AI system's prediction about review authenticity.
Return answers EXACTLY with the following 4 lines (no extra text):
Cross-Check Result: [AGREE or DISAGREE]
Final Classification: [LEGITIMATE or FAKE]
Confidence: [0-100]
Reasoning: [short reason]

REVIEW TO VALIDATE:
Text: "{review_data['text']}"
Reviewer: {review_data['review_count']} total reviews, Local Guide: {"Yes" if review_data['is_local_guide'] else "No"}
Posted: {review_data['timestamp'].strftime('%Y-%m-%d %H:%M (%A)')}

AI SYSTEM'S PREDICTION:
Classification: {ml_prediction['prediction'].upper()}
Confidence: {ml_prediction['confidence_fake']:.2f}

Indicators of FAKE:
- Promotional content or contact info (e.g., 'call now', 'www.')
- Generic superlatives ('best ever', 'amazing', 'perfect') and exclamation spam
- No specific details of a real visit
- Very new reviewer with overly positive language

Indicators of LEGITIMATE:
- Specific details about the visit (what, when, who)
- Balanced or nuanced language
- Natural phrasing

Examples:
FAKE: "Amazing restaurant! Best food ever! Call us at 65840111!"
LEGITIMATE: "Good noodles; broth aromatic but a bit oily. Staff were friendly but slow at lunch."

Now decide whether to AGREE or DISAGREE with the AI prediction, and provide your own classification with a confidence.
"""
        return prompt
    
    def query_llm_cross_check(self, prompt: str) -> Dict:
        """Send cross-check prompt to Llama"""
        
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'num_predict': 200,
                'top_k': 10,
                'top_p': 0.3
            }
        }
        
        try:
            response = self.session.post(self.base_url, json=payload, timeout=45)
            
            if response.status_code == 200:
                content = response.json().get('response', '')
                parsed = self._parse_cross_check_response(content)
                parsed['success'] = True
                return parsed
            else:
                return {
                    'success': False,
                    'cross_check_result': 'UNCERTAIN',
                    'final_classification': 'UNCERTAIN',
                    'confidence': 50,
                    'reasoning': 'API error'
                }
                
        except Exception as e:
            return {
                'success': False,
                'cross_check_result': 'UNCERTAIN',
                'final_classification': 'UNCERTAIN',
                'confidence': 50,
                'reasoning': f'Error: {str(e)}'
            }
    
    def _parse_cross_check_response(self, response: str) -> Dict:
        """Parse LLM cross-check response"""
        import re
        
        result = {
            'cross_check_result': 'UNCERTAIN',
            'final_classification': 'UNCERTAIN', 
            'confidence': 50,
            'reasoning': 'Could not parse response'
        }
        
        try:
            # Extract cross-check result
            cross_check_match = re.search(r'Cross-Check Result:\s*(AGREE|DISAGREE|UNCERTAIN)', response, re.IGNORECASE)
            if cross_check_match:
                result['cross_check_result'] = cross_check_match.group(1).upper()
            
            # Extract final classification
            classification_match = re.search(r'Final Classification:\s*(LEGITIMATE|FAKE)', response, re.IGNORECASE)
            if classification_match:
                result['final_classification'] = classification_match.group(1).upper()
            
            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*(\d+)', response)
            if confidence_match:
                result['confidence'] = int(confidence_match.group(1))
                result['confidence'] = max(0, min(100, result['confidence']))
            
            # Extract reasoning
            reasoning_match = re.search(r'Reasoning:\s*([^\n]+)', response)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()[:200]
        
        except Exception as e:
            result['reasoning'] = f'Parse error: {str(e)}'
        
        return result
    
    def _should_trigger_cross_check(self, ml_label: str, p_fake: float) -> bool:
        """Decide whether to cross-check, using an uncertainty band plus class-specific rules."""
        if abs(p_fake - 0.5) < self.confidence_threshold:
            return True
        # check borderline cases asymmetrically
        if ml_label == 'fake' and p_fake < 0.55:
            return True
        if ml_label == 'legitimate' and p_fake > 0.45:
            return True
        return False
    
    def _fuse_predictions(self, ml_label: str, p_fake: float, llm_label: str, llm_conf: int) -> str:
        """Asymmetric fusion: prefer catching fakes, protect precision with stronger bar for flipping to legit."""
        # Case 1: ML says LEGIT, LLM confident it's FAKE -> flip to fake
        if ml_label == 'legitimate' and llm_label == 'FAKE' and llm_conf >= 70:
            return 'fake'
        # Case 2: ML says FAKE, LLM strongly says LEGIT and ML fake prob is weak -> flip to legit
        if ml_label == 'fake' and llm_label == 'LEGITIMATE' and llm_conf >= 80 and p_fake < 0.55:
            return 'legitimate'
        # Otherwise, keep ML label
        return ml_label
    
    def predict_with_cross_validation(self, text: str, is_local_guide: bool, 
                                    timestamp: datetime, review_count: int) -> Dict:
        """Two-stage prediction: ML model + optional LLM cross-check"""
        
        ml_prediction = self.ml_model.predict_single_review(
            text=text,
            is_local_guide=is_local_guide,
            timestamp=timestamp,
            review_count=review_count
        )
        
        result = {
            'ml_prediction': ml_prediction,
            'llm_cross_check': None,
            'final_decision': ml_prediction['prediction'],
            'final_confidence': ml_prediction['confidence_fake'],
            'cross_check_triggered': False,
            'agreement_status': 'N/A'
        }
        
        self.stats['total_predictions'] += 1
        
        ml_label = ml_prediction['prediction']
        p_fake = ml_prediction['confidence_fake']
        
        needs_cross_check = self._should_trigger_cross_check(ml_label, p_fake)
        
        if needs_cross_check:
            review_data = {
                'text': text,
                'is_local_guide': is_local_guide,
                'timestamp': timestamp,
                'review_count': review_count
            }
            
            prompt = self.create_cross_check_prompt(review_data, ml_prediction)
            llm_result = self.query_llm_cross_check(prompt)
            
            result['llm_cross_check'] = llm_result
            result['cross_check_triggered'] = True
            self.stats['cross_checks_performed'] += 1
            
            if llm_result['success']:
                llm_label = llm_result['final_classification']
                llm_conf = llm_result['confidence']
                
                if llm_result['cross_check_result'] == 'AGREE':
                    result['agreement_status'] = 'AGREE'
                    #confidence bump
                    result['final_confidence'] = min(max(p_fake, llm_conf / 100.0), 1.0)
                    self.stats['agreements'] += 1
                
                elif llm_result['cross_check_result'] == 'DISAGREE':
                    result['agreement_status'] = 'DISAGREE'
                    fused_label = self._fuse_predictions(ml_label, p_fake, llm_label, llm_conf)
                    result['final_decision'] = fused_label
                    result['final_confidence'] = llm_conf / 100.0 if fused_label != ml_label else p_fake
                    self.stats['disagreements'] += 1
                    if fused_label == 'fake' and ml_label != 'fake':
                        self.stats['flips_to_fake'] += 1
                    if fused_label == 'legitimate' and ml_label != 'legitimate':
                        self.stats['flips_to_legit'] += 1
                
                else:  # UNCERTAIN
                    result['agreement_status'] = 'UNCERTAIN'
                    result['final_confidence'] = min(p_fake, llm_conf / 100.0)
            # if not success: keep ML
        return result
