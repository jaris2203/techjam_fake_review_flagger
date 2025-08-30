import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import random
import re

class ReviewAuthenticityDetector:
    """Google Maps Review Authenticity Detection System"""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        
        self.policies = {
            'no_advertisements': r'(call now|visit our|contact us|phone|website|www\.|\.com|discount|offer|sale)',
            'no_promotional': r'(best ever|amazing|perfect|outstanding|incredible|life changing)',
            'no_irrelevant': r'(unrelated|off topic|random)',
            'generic_patterns': r'(highly recommend|will come back|5 stars|excellent service)'
        }
    
    def extract_columns_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Extract and clean required columns from CSV file"""
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = ['isLocalGuide', 'publishedAtDate', 'reviewerNumberOfReviews', 'text', 'originalLanguage']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None
        
        # Extract and filter for English only
        df_clean = df[required_columns].copy()
        df_clean = df_clean[df_clean['originalLanguage'] == 'en']
        df_clean = df_clean.dropna()
        
        # Rename columns for consistency
        df_clean.columns = ['is_local_guide', 'timestamp', 'review_count', 'text', 'original_language']
        
        # Convert timestamp
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        
        # Add label column if it exists in original data
        if 'label' in df.columns:
            df_clean['label'] = df.loc[df_clean.index, 'label']
        
        return df_clean
    
    def generate_synthetic_fake_reviews(self, legitimate_df: pd.DataFrame, num_fakes: int = 100) -> pd.DataFrame:
        """Generate synthetic fake reviews to balance the dataset"""
        
        fake_patterns = [
            "Amazing {business_type}! Best {service} ever! 5 stars!",
            "This place changed my life. Perfect in every way. Highly recommend!",
            "Outstanding {business_type}. Excellent service and quality. Will definitely come back!",
            "Incredible experience! The {service} was absolutely perfect. Amazing staff!",
            "Best {business_type} in town! Everything was flawless. 10/10 would recommend!",
            "Call now for best deals! Visit our website www.example.com! Amazing discounts!",
            "Check out our amazing {service}! Contact us today for special offers!",
            "Good place. Nice {service}. Recommended.",
            "OK {business_type}. Average {service}. Nothing special.",
        ]
        
        business_types = ['restaurant', 'cafe', 'hotel', 'shop', 'store', 'place']
        services = ['food', 'service', 'experience', 'atmosphere', 'quality', 'staff']
        
        synthetic_fakes = []
        
        for i in range(num_fakes):
            pattern = random.choice(fake_patterns)
            business_type = random.choice(business_types)
            service = random.choice(services)
            
            fake_text = pattern.format(business_type=business_type, service=service)
            
            fake_review = {
                'is_local_guide': random.choice([False, False, False, True]),
                'timestamp': self._generate_suspicious_timestamp(),
                'review_count': random.choices([1, 2, 3, 4, 5, 10, 20], weights=[30, 25, 20, 15, 5, 3, 2])[0],
                'text': fake_text,
                'original_language': 'en',
                'label': 'fake'
            }
            synthetic_fakes.append(fake_review)
        
        return pd.DataFrame(synthetic_fakes)
    
    def _generate_suspicious_timestamp(self) -> datetime:
        """Generate timestamps that might be suspicious"""
        from datetime import timedelta
        base_date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        if random.random() < 0.4:  # 40% suspicious timing
            hour = random.choice(list(range(0, 6)) + list(range(23, 24)))
        else:
            hour = random.randint(8, 22)
        
        return base_date.replace(hour=hour, minute=random.randint(0, 59))
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from review data"""
        
        features_list = []
        
        for _, row in df.iterrows():
            text = str(row['text']).lower()
            text_length = len(text)
            word_count = len(text.split())
            exclamation_count = text.count('!')
            caps_ratio = sum(1 for c in row['text'] if c.isupper()) / max(len(row['text']), 1)
            
            # Policy violation features
            policy_violations = self._check_policy_violations(text)
            
            # Reviewer features
            review_count_log = np.log(row['review_count'] + 1)
            is_local_guide = int(row['is_local_guide'])
            
            # Temporal features
            hour = row['timestamp'].hour
            is_weekend = row['timestamp'].weekday() >= 5
            is_off_hours = hour < 6 or hour > 22
            
            # Credibility score
            credibility = self._calculate_credibility(row['review_count'], row['is_local_guide'])
            
            feature_vector = [
                text_length, word_count, exclamation_count, caps_ratio,
                review_count_log, is_local_guide, hour, int(is_weekend), 
                int(is_off_hours), credibility
            ] + list(policy_violations.values())
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def _check_policy_violations(self, text: str) -> dict:
        """Check for policy violations in text"""
        violations = {}
        for policy, pattern in self.policies.items():
            violations[f'{policy}_violation'] = int(bool(re.search(pattern, text, re.IGNORECASE)))
        return violations
    
    def _calculate_credibility(self, review_count: int, is_local_guide: bool) -> float:
        """Calculate reviewer credibility score"""
        count_score = min(review_count / 50, 1.0)
        guide_bonus = 0.3 if is_local_guide else 0.0
        return count_score + guide_bonus
    
    def train_model(self, df: pd.DataFrame) -> dict:
        """Train the authenticity detection model"""
        
        # Prepare features and labels
        X_text = df['text'].values
        X_features = self.extract_features(df)
        y = (df['label'] == 'fake').astype(int)
        
        # Vectorize text
        X_text_vectorized = self.text_vectorizer.fit_transform(X_text)
        
        # Combine features
        X_combined = np.hstack([X_text_vectorized.toarray(), self.scaler.fit_transform(X_features)])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        return {'metrics': metrics}
    
    def predict_single_review(self, text: str, is_local_guide: bool, timestamp: datetime, review_count: int) -> dict:
        """Predict authenticity of a single review"""
        
        # Create temporary DataFrame
        temp_df = pd.DataFrame([{
            'text': text,
            'is_local_guide': is_local_guide,
            'timestamp': timestamp,
            'review_count': review_count,
            'original_language': 'en',
            'label': 'unknown'
        }])
        
        # Extract features
        X_text = self.text_vectorizer.transform([text])
        X_features = self.scaler.transform(self.extract_features(temp_df))
        X_combined = np.hstack([X_text.toarray(), X_features])
        
        # Predict
        prediction = self.model.predict(X_combined)[0]
        confidence = self.model.predict_proba(X_combined)[0]
        
        # Policy violations
        policy_violations = self._check_policy_violations(text.lower())
        
        return {
            'prediction': 'fake' if prediction == 1 else 'legitimate',
            'confidence_fake': confidence[1],
            'confidence_legitimate': confidence[0],
            'policy_violations': policy_violations,
            'credibility_score': self._calculate_credibility(review_count, is_local_guide)
        }