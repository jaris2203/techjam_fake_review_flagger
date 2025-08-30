
import pandas as pd
import os
from datetime import datetime
import math

def main():
    """Main pipeline with three-model evaluation and class-balance safeguards."""

    CONFIG = {
        'training_csv': 'data/training_reviews.csv',
        'pseudo_csv': 'data/pseudo_reviews.csv',
        'test_csv': 'data/test_reviews.csv',
        'output_dir': 'results',
        'pseudo_confidence': 0.75,
        'cross_check_confidence': 0.3,
        'target_fake_ratio': 0.45
    }

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    try:
        from review_authenticity_system import ReviewAuthenticityDetector
        from llama_setup import LlamaOptimizedPseudoLabeler  
        from dual_validation_system import DualValidationSystem
        from evaluation import EvaluationSystem
    except ImportError as e:
        print(f"Missing module: {e}")
        print("Required files: review_authenticity_system.py, llama_setup.py, dual_validation_system.py, evaluation.py")
        return None

    evaluator = EvaluationSystem(output_dir=CONFIG['output_dir'])

    test_df = evaluator.load_test_set(CONFIG['test_csv'])
    if test_df is None:
        print("Failed to load test set")
        return None
    print(f"Loaded {len(test_df)} test reviews (never used for training)")

    detector = ReviewAuthenticityDetector()
    training_df = detector.extract_columns_from_csv(CONFIG['training_csv'])
    print(f"Loaded {len(training_df)} training reviews")

    # Balance baseline set to target fake ratio
    if 'label' in training_df.columns:
        total = len(training_df)
        fake_count = int((training_df['label'] == 'fake').sum())
        target = CONFIG['target_fake_ratio']
        need = (target * total) - fake_count
        if need > 0:
            # Solve n fakes such that (fake_count + n)/(total + n) >= target
            n = math.ceil(need / (1 - target))
            extra_fakes = detector.generate_synthetic_fake_reviews(training_df, num_fakes=n)
            baseline_training_df = pd.concat([training_df, extra_fakes], ignore_index=True)
            print(f"Baseline balancing: added {len(extra_fakes)} synthetic fakes; total {len(baseline_training_df)} reviews")
        else:
            baseline_training_df = training_df.copy()
    else:
        baseline_training_df = training_df.copy()

    baseline_results = detector.train_model(baseline_training_df)
    # PHASE 3: Enhanced model with pseudo-labeling
    enhanced_detector = None
    enhanced_eval = None
    if os.path.exists(CONFIG['pseudo_csv']):
        unlabeled_df = detector.extract_columns_from_csv(CONFIG['pseudo_csv'])
        if unlabeled_df is not None:
            unlabeled_df = unlabeled_df.head(50)
            if 'label' in unlabeled_df.columns:
                unlabeled_df = unlabeled_df.drop(columns=['label'])
            print(f"Loaded {len(unlabeled_df)} unlabeled reviews for pseudo-labeling")

            pseudo_labeler = LlamaOptimizedPseudoLabeler(confidence_threshold=CONFIG['pseudo_confidence'])
            all_pseudo, high_conf_pseudo = pseudo_labeler.generate_pseudo_labels_optimized(unlabeled_df, batch_size=5)

            pseudo_samples = []
            for _, pseudo_row in high_conf_pseudo.iterrows():
                if pseudo_row['success'] and pseudo_row['classification'] in ['LEGITIMATE', 'FAKE']:
                    orig_idx = pseudo_row['original_index']
                    if orig_idx < len(unlabeled_df):
                        orig_review = unlabeled_df.iloc[orig_idx]
                        pseudo_samples.append({
                            'is_local_guide': orig_review['is_local_guide'],
                            'timestamp': orig_review['timestamp'],
                            'review_count': orig_review['review_count'],
                            'text': orig_review['text'],
                            'original_language': 'en',
                            'label': 'fake' if pseudo_row['classification'] == 'FAKE' else 'legitimate'
                        })

            if len(pseudo_samples) > 0:
                pseudo_df = pd.DataFrame(pseudo_samples)
                enhanced_training_df = pd.concat([baseline_training_df, pseudo_df], ignore_index=True)

                # Rebalance enhanced set to target fake ratio directly
                total = len(enhanced_training_df)
                fake_count = int((enhanced_training_df['label'] == 'fake').sum())
                target = CONFIG['target_fake_ratio']
                need = (target * total) - fake_count
                if need > 0:
                    n = math.ceil(need / (1 - target))
                    extra_fakes = detector.generate_synthetic_fake_reviews(training_df, num_fakes=n)
                    enhanced_training_df = pd.concat([enhanced_training_df, extra_fakes], ignore_index=True)
                    print(f"Enhanced balancing: added {len(extra_fakes)} synthetic fakes; total {len(enhanced_training_df)} reviews")

                enhanced_detector = ReviewAuthenticityDetector()
                enhanced_results = enhanced_detector.train_model(enhanced_training_df)
            else:
                return
        else:
            return 
            #
    else:
       return

    # PHASE 4: Evaluate on independent test set
    evaluation_results = []
    baseline_eval = evaluator.evaluate_model_performance(
        model=detector,
        test_df=test_df,
        model_name="Baseline Model"
    )
    evaluation_results.append(baseline_eval)

    if enhanced_detector:
        enhanced_eval = evaluator.evaluate_model_performance(
            model=enhanced_detector,
            test_df=test_df,
            model_name="Enhanced Model (with Pseudo-labels)"
        )
        evaluation_results.append(enhanced_eval)

    best_model = enhanced_detector if enhanced_detector else detector
    dual_validator = DualValidationSystem(
        trained_model=best_model,
        confidence_threshold=CONFIG['cross_check_confidence']
    )
    dual_eval = evaluator.evaluate_dual_validation_system(
        dual_validator=dual_validator,
        test_df=test_df
    )

    # PHASE 5: Comparison and results
    all_results = evaluation_results + [dual_eval]
    comparison = evaluator.compare_models(all_results)

    final_results = {
        'config': CONFIG,
        'test_set_size': len(test_df),
        'baseline_evaluation': baseline_eval,
        'enhanced_evaluation': enhanced_eval if enhanced_detector else None,
        'dual_validation_evaluation': dual_eval,
        'model_comparison': comparison,
        'timestamp': datetime.now().isoformat()
    }

    results_file = evaluator.save_results(final_results, f"{CONFIG['output_dir']}/complete_evaluation.json")
    return final_results

if __name__ == "__main__":
    results = main()
