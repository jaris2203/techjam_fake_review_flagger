import pandas as pd
import requests
import time
from datetime import datetime
from tqdm import tqdm

class LlamaOptimizedPseudoLabeler:
    """Llama-based pseudo-labeling system for review authenticity"""

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        self.base_url = "http://localhost:11434/api/generate"
        self.model_name = "llama2"
        self.session = requests.Session()

    def check_llama_availability(self) -> bool:
        """Check if Llama/Ollama is running"""
        try:
            response = self.session.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                return self.model_name in available_models
            return False
        except:
            return False

    def create_pseudo_labeling_prompt(self, review_data: dict) -> str:
        """Create optimized prompt for Llama pseudo-labeling"""

        prompt = f"""Task: Classify this Google Maps review as LEGITIMATE or FAKE.

Review: "{review_data['text']}"
Reviewer: {review_data['review_count']} total reviews, Local Guide: {"Yes" if review_data['is_local_guide'] else "No"}
Posted: {review_data['timestamp'].strftime('%H:%M on %A')}

FAKE indicators:
- Generic phrases: "amazing", "best ever", "perfect"
- No specific details about experience
- Promotional content or contact info
- New reviewer with overly positive language

LEGITIMATE indicators:
- Specific details about visit
- Balanced view (pros and cons)
- Natural language, personal observations

Examples:
FAKE: "Amazing place! Best food ever! 5 stars!"
LEGITIMATE: "Good pizza. Fresh ingredients but service was slow during lunch."

Answer format:
Classification: LEGITIMATE or FAKE
Confidence: 0-100
Reason: Brief explanation

Classification:"""

        return prompt

    def query_llama_single(self, prompt: str, timeout: int = 30) -> dict:

        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'top_k': 10,
                'top_p': 0.3,
                'num_predict': 100,
                'stop': ['\n\n', 'Human:', 'User:']
            }
        }

        try:
            response = self.session.post(self.base_url, json=payload, timeout=timeout)

            if response.status_code == 200:
                content = response.json().get('response', '').strip()
                parsed = self._parse_llama_response(content)

                return {
                    'success': True,
                    'classification': parsed['classification'],
                    'confidence': parsed['confidence'],
                    'reasoning': parsed['reasoning'],
                    'raw_response': content
                }
            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'classification': 'UNCERTAIN',
                    'confidence': 50,
                    'reasoning': 'API error'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'classification': 'UNCERTAIN',
                'confidence': 50,
                'reasoning': f'Error: {str(e)}'
            }

    def _parse_llama_response(self, response: str) -> dict:
        """Parse Llama response to extract classification and confidence"""
        import re

        classification = 'UNCERTAIN'
        confidence = 50
        reasoning = 'Could not parse response'

        try:
            # Classification
            if re.search(r'\b(LEGITIMATE|LEGIT)\b', response, re.IGNORECASE):
                classification = 'LEGITIMATE'
            elif re.search(r'\b(FAKE|FALSE)\b', response, re.IGNORECASE):
                classification = 'FAKE'

            # Confidence
            conf_patterns = [
                r'Confidence[:\s]*(\d+)',
                r'(\d+)%',
                r'(\d+)/100',
                r'score[:\s]*(\d+)'
            ]
            for pattern in conf_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    confidence = int(match.group(1))
                    confidence = max(0, min(100, confidence))
                    break

            # Reasoning
            reason_patterns = [
                r'Reason[:\s]*([^\n]+)',
                r'because[:\s]*([^\n]+)',
                r'(Generic|Specific|Promotional|Natural)[^\.]*\.'
            ]
            for pattern in reason_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    reasoning = match.group(1).strip()[:100]
                    break

            if reasoning == 'Could not parse response' and len(response) > 10:
                reasoning = response[:80] + "..." if len(response) > 80 else response

        except Exception as e:
            reasoning = f"Parse error: {str(e)[:50]}"

        return {
            'classification': classification,
            'confidence': confidence,
            'reasoning': reasoning
        }

    def generate_pseudo_labels_optimized(self, unlabeled_df: pd.DataFrame, batch_size: int = 5) -> tuple:
        """Generate pseudo-labels for unlabeled reviews"""

        results = []

        with tqdm(total=len(unlabeled_df), desc="Pseudo-labeling") as pbar:
            for idx, (_, row) in enumerate(unlabeled_df.iterrows()):

                review_data = {
                    'text': row['text'],
                    'review_count': row['review_count'],
                    'is_local_guide': row['is_local_guide'],
                    'timestamp': row['timestamp']
                }

                prompt = self.create_pseudo_labeling_prompt(review_data)
                result = self.query_llama_single(prompt)
                result['original_index'] = row.name

                results.append(result)
                pbar.update(1)

                # Pause between batches
                if (idx + 1) % batch_size == 0:
                    time.sleep(1)

        pseudo_labels_df = pd.DataFrame(results)

        # Filter high confidence results
        high_confidence_mask = (
            (pseudo_labels_df['success'] == True) & 
            (pseudo_labels_df['confidence'] >= self.confidence_threshold * 100) &
            (pseudo_labels_df['classification'].isin(['LEGITIMATE', 'FAKE']))
        )

        high_confidence_df = pseudo_labels_df[high_confidence_mask].copy()

        return pseudo_labels_df, high_confidence_df
