# Review Authenticity & Relevancy Detection (Hackathon Project)

##  Overview
Problem statement chosen: Design and implement an ML-based system to evaluate the quality and relevancy of Google location reviews

This project builds a **Dual Validation pipeline** to detect low-quality, spammy, or irrelevant Google Maps reviews.  
It combines **Machine Learning** for efficiency and **LLM cross-checks** for nuanced reasoning.  
Key goals:
- Flag fake, promotional, or off-topic reviews
- Ensure reviews are genuinely relevant to the location
- Provide robust performance even when pseudo-labeling is noisy

---

##  Features
- **Baseline ML Model**
  - Random Forest with TF-IDF + engineered metadata features
  - Policy violation detection (regex for ads, URLs, promotions)
  - Synthetic fake review generation for class balancing
- **Pseudo-Labeling**
  - Uses **Llama 2** via Ollama API (`http://localhost:11434/api/generate`)
  - Hard prompting: forces structured output (label, confidence, reasoning)
  - Expands training dataset with high-confidence pseudo-labeled reviews
- **Dual Validation System**
  - ML makes initial prediction
  - If prediction is uncertain, trigger LLM cross-check
  - Asymmetric fusion rules (prefer recall on fakes, protect precision on legits)

---
## 🛠️ Tech Stack

- **Programming**: Python 3.10  
- **Data Processing**: pandas, NumPy  
- **ML**: scikit-learn, RandomForestClassifier  
- **NLP**: Hugging Face Transformers, spaCy, NLTK, Gensim  
- **Vectorization**: TfidfVectorizer  
- **Visualization**: matplotlib, seaborn, Plotly  
- **APIs**:  
  - **Ollama / Llama 2 API** for pseudo-labeling & validation  
  - **Google Maps Reviews dataset** as data source  
- **Environment**: VSCode

---

## Results
On the held-out test set:

| Model                          | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|--------------------------------|----------|-----------|--------|----------|---------|
| Baseline Model                 | 0.833    | 0.833     | 0.500  | 0.625    | 0.969   |
| Enhanced Model (Pseudo-labels) | 0.722    | 0.000     | 0.000  | 0.000    | 0.919   |
| Dual Validation System         | 0.750    | 0.571     | 0.400  | 0.471    |   NaN   |

- **Baseline**: solid starting point  
- **Enhanced**: collapsed due to biased pseudo-labels → highlights the risk of semi-supervised learning  
- **Dual Validation**: recovered performance by flipping uncertain cases using LLM cross-checks  


### Key Insights
- **Class imbalance is critical**: Without balancing, the model learns to predict “legitimate” by default. Synthetic fakes were necessary.  
- **Pseudo-labeling can backfire**: More data isn’t always better — pseudo-labels dominated by “legitimate” diluted the signal and hurt performance.  
- **LLM cross-checking adds robustness**: Dual Validation did not outperform the baseline, but it prevented a total collapse, ensuring the system still catches fakes even when pseudo-labeling fails.  
- **Asymmetric fusion rules matter**: Allowing the LLM to upgrade uncertain “legitimate” predictions to “fake,” but requiring higher confidence to downgrade, provided a good trade-off between recall and precision.

---

### Limitations
1. **Pseudo-label quality**  
   - Most LLM pseudo-labels came back as “legitimate,” providing little new information.  
   - As a result, Enhanced training data was biased, leading to collapse.

2. **Synthetic fake generation**  
   - Current patterns are obvious (“Call now!”, “Best ever!”).  
   - While useful for balancing, they may not reflect realistic modern spam/fake reviews.

3. **Threshold selection**  
   - Small validation splits sometimes led to extreme thresholds (predicting all legit).  
   - Adaptive fallback rules were added, but results are still sensitive to dataset size.

4. **Dual Validation overhead**  
   - Requires running an LLM (via Ollama / Llama 2).  
   - This adds latency and compute costs, making it less scalable in production.

5. **Dataset size**  
   - Relatively small labeled set → results are volatile. A few misclassifications can change F1 dramatically.  
   - With larger labeled data, Enhanced + Dual Validation would likely shine more.

---

### Future Work
- **Improve pseudo-labeling**: Use active learning to request labels on most uncertain reviews instead of blindly pseudo-labeling.  
- **Harder synthetic fakes**: Generate subtle spammy reviews (e.g., sentiment-biased but irrelevant) to stress-test the model.  
- **Confidence calibration**: Use Platt scaling or isotonic regression to better align probability outputs with real uncertainty.  
- **Broader evaluation**: Test on larger, more diverse review datasets to validate generalization.  
- **Optimize Dual Validation**: Explore lighter LLMs or distilled models for faster cross-checking.
---

## Project Structure
├── main.py # Orchestration script (baseline, enhanced, dual validation)
├── review_authenticity_system.py # ML model, feature engineering, synthetic fake generation
├── dual_validation_system.py # ML + LLM cross-validation logic
├── llama_setup.py # Hard-prompt setup for Llama pseudo-labeling
├── evaluation.py # Evaluation metrics, confusion matrices
├── results/ # Saved metrics and plots
└── README.md

## How to Reproduce the Results

**Clone this repo and install dependencies**
   ```bash
   git clone https://github.com/jaris2203/techjam_fake_review_flagger.git
   cd techjam_fake_review_flagger
   pip install -r requirements.txt
   ```
   Download and Install Ollama from https://ollama.com/download/windows

   ```bash
   ollama serve
   ollama pull llama2
   python main.py
   ```
