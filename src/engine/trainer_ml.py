import time
import logging
from sklearn.metrics import accuracy_score, classification_report, f1_score

log = logging.getLogger(__name__)

def train_and_evaluate_ml(pipeline, X_train, y_train, X_test, y_test, callbacks=None):
    """Executes the standard fit/predict cycle for Classical ML models."""
    if callbacks is None:
        callbacks = []
        
    for cb in callbacks: cb.on_train_begin(0, 0)
        
    log.info("Starting Classical ML pipeline training...")
    start_time = time.time()
    
    pipeline.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    log.info(f"Training completed in {train_time:.2f} seconds.")
    
    log.info("Evaluating on test set...")
    inf_start = time.time()
    preds = pipeline.predict(X_test)
    inf_time = time.time() - inf_start
    
    acc = accuracy_score(y_test, preds)
    acc_percent = acc * 100
    macro_f1 = f1_score(y_test, preds, average='macro')
    ms_per_image = (inf_time / len(X_test)) * 1000.0
    
    log.info(f"Test Accuracy: {acc_percent:.2f}% | F1: {macro_f1:.4f} | Latency: {ms_per_image:.4f} ms/img")
    log.info(f"\n{classification_report(y_test, preds, zero_division=0)}")
    
    for cb in callbacks: cb.on_train_end(test_accuracy=acc_percent, f1_score=macro_f1, inference_latency_ms=ms_per_image)
    
    return acc