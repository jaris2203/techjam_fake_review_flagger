import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
from datetime import datetime
import os

class EvaluationSystem:
    """Evaluation system using independent test set"""

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_test_set(self, test_csv_path: str) -> pd.DataFrame:
        """Load and validate the independent test set"""

        from review_authenticity_system import ReviewAuthenticityDetector
        detector = ReviewAuthenticityDetector()

        # Load and clean test data
        test_df = detector.extract_columns_from_csv(test_csv_path)

        if test_df is None:
            return None

        # Validate test set has labels
        if 'label' not in test_df.columns:
            print("Test set must have a 'label' column with 'legitimate' or 'fake' values.")
            return None

        label_counts = test_df['label'].value_counts()
        print("Test Set Composition:")
        for label, count in label_counts.items():
            print(f"   {label.capitalize()}: {count} ({count/len(test_df)*100:.1f}%)")

        return test_df

    def evaluate_model_performance(self, model, test_df: pd.DataFrame, model_name: str = "Model") -> dict:
        """Comprehensive evaluation of a model on the test set"""

        predictions = []
        true_labels = []
        confidence_scores = []

        for _, row in test_df.iterrows():
            try:
                pred_result = model.predict_single_review(
                    text=row['text'],
                    is_local_guide=row['is_local_guide'],
                    timestamp=row['timestamp'],
                    review_count=row['review_count']
                )

                predictions.append(1 if pred_result['prediction'] == 'fake' else 0)
                confidence_scores.append(pred_result['confidence_fake'])
                true_labels.append(1 if row['label'] == 'fake' else 0)

            except Exception as e:
                print(f"Error predicting sample: {e}")
                predictions.append(0)
                confidence_scores.append(0.5)
                true_labels.append(1 if row['label'] == 'fake' else 0)

        # Convert to numpy arrays
        y_true = np.array(true_labels)
        y_pred = np.array(predictions)
        y_prob = np.array(confidence_scores)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        results = {
            'model_name': model_name,
            'test_set_size': len(test_df),
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'evaluation_timestamp': datetime.now().isoformat()
        }

        self._print_evaluation_results(results)
        return results

    def evaluate_dual_validation_system(self, dual_validator, test_df: pd.DataFrame) -> dict:
        """Evaluate dual validation system on the test set"""

        dual_predictions = []
        ml_predictions = []
        true_labels = []
        cross_check_info = []

        for _, row in test_df.iterrows():
            try:
                dual_result = dual_validator.predict_with_cross_validation(
                    text=row['text'],
                    is_local_guide=row['is_local_guide'],
                    timestamp=row['timestamp'],
                    review_count=row['review_count']
                )

                dual_predictions.append(1 if dual_result['final_decision'] == 'fake' else 0)
                ml_predictions.append(1 if dual_result['ml_prediction']['prediction'] == 'fake' else 0)
                true_labels.append(1 if row['label'] == 'fake' else 0)

                cross_check_info.append({
                    'triggered': dual_result['cross_check_triggered'],
                    'agreement': dual_result['agreement_status']
                })

            except Exception as e:
                print(f"Error in dual validation: {e}")
                dual_predictions.append(0)
                ml_predictions.append(0)
                true_labels.append(1 if row['label'] == 'fake' else 0)
                cross_check_info.append({'triggered': False, 'error': str(e)})

        # Convert to numpy arrays
        y_true = np.array(true_labels)
        y_pred_dual = np.array(dual_predictions)
        y_pred_ml = np.array(ml_predictions)

        # Calculate metrics for both
        dual_metrics = {
            'accuracy': accuracy_score(y_true, y_pred_dual),
            'precision': precision_score(y_true, y_pred_dual, zero_division=0),
            'recall': recall_score(y_true, y_pred_dual, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_dual, zero_division=0)
        }

        ml_only_metrics = {
            'accuracy': accuracy_score(y_true, y_pred_ml),
            'precision': precision_score(y_true, y_pred_ml, zero_division=0),
            'recall': recall_score(y_true, y_pred_ml, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_ml, zero_division=0)
        }

        # Calculate improvement
        improvement = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            dual_val = dual_metrics[metric]
            ml_val = ml_only_metrics[metric]
            improvement[metric] = {
                'absolute': dual_val - ml_val,
                'relative_pct': ((dual_val - ml_val) / ml_val * 100) if ml_val > 0 else 0
            }

        # Cross-check stats
        cross_checks_performed = sum(1 for cc in cross_check_info if cc.get('triggered', False))
        agreements = sum(1 for cc in cross_check_info if cc.get('agreement') == 'AGREE')

        results = {
            'model_name': 'Dual Validation System',
            'test_set_size': len(test_df),
            'dual_validation_metrics': dual_metrics,
            'ml_only_metrics': ml_only_metrics,
            'improvement': improvement,
            'cross_checks_performed': cross_checks_performed,
            'agreement_rate': agreements / max(1, cross_checks_performed),
            'evaluation_timestamp': datetime.now().isoformat()
        }

        self._print_dual_validation_results(results)
        return results

    def _print_evaluation_results(self, results: dict):
        metrics = results['metrics']

        # print(f"\n{results['model_name']} Performance:")
        # print("=" * 40)
        # print(f"Accuracy:     {metrics['accuracy']:.3f}")
        # print(f"Precision:    {metrics['precision']:.3f}")
        # print(f"Recall:       {metrics['recall']:.3f}")
        # print(f"F1-Score:     {metrics['f1_score']:.3f}")
        # print(f"AUC-ROC:      {metrics['auc_roc']:.3f}")

        # # Confusion Matrix
        # cm = np.array(results['confusion_matrix'])
        # print("\nConfusion Matrix:")
        # print("                 Predicted")
        # print("                 Legit  Fake")
        # print(f"Actual Legit     {cm[0][0]:4d}  {cm[0][1]:4d}")
        # print(f"       Fake      {cm[1][0]:4d}  {cm[1][1]:4d}")

    def _print_dual_validation_results(self, results: dict):
        print()

    def save_results(self, results: dict, filename: str = None):
        """Save evaluation results to JSON file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/evaluation_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to: {filename}")
        return filename

    def compare_models(self, results_list: list) -> dict:
        """Compare multiple model evaluation results"""

        print("\nModel Comparison Summary:")
        print("=" * 50)

        comparison_data = []
        for result in results_list:
            model_data = {
                'model_name': result['model_name'],
                **result.get('metrics', result.get('dual_validation_metrics', {}))
            }
            comparison_data.append(model_data)

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.3f'))

        return {'comparison_table': comparison_df.to_dict('records')}
