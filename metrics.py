import numpy as np
from sklearn.metrics import roc_curve

def compute_eer_threshold(y_true, y_scores):
    """
    Calcule le seuil qui donne l'Equal Error Rate (EER)
    
    Args:
        y_true (numpy vector): Vecteur des labels (0 ou 1), de taille (N,)
        y_scores (numpy vector): Vecteur des scores prédits, de taille (N,)

    Returns:
        threshold (float): Seuil optimal basé sur l'EER
    """
    # Assure toi que c'est sur CPU et en numpy pour sklearn
    y_true_np = y_true  # .detach().cpu().numpy()
    y_scores_np = y_scores  #.detach().cpu().numpy()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np)
    
    # Trouver l'endroit où |FPR - (1-TPR)| est minimal (Equal Error Rate)
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eer_idx]
    
    return eer_threshold

def compute_accuracy_with_eer_threshold(y_true, y_scores, all_metrics=False):
    """
    Calcule l'accuracy basée sur le seuil EER

    Args:
        y_true (numpy vector): Vecteur des labels (0 ou 1), de taille (N,)
        y_scores (numpy vector): Vecteur des scores prédits, de taille (N,)

    Returns:
        accuracy (float): L'accuracy obtenue en appliquant le seuil EER
        threshold (float): Le seuil utilisé
    """
    threshold = compute_eer_threshold(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(np.int16)
    
    accuracy = np.mean(y_pred == y_true)
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    if all_metrics:
        return accuracy, threshold, precision, recall, f1_score
    else:    
        return accuracy, threshold


def compute_precision_recall(y_true, y_scores):
    """
    Calcule la précision et le rappel à partir des scores prédits

    Args:
        y_true (numpy vector): Vecteur des labels (0 ou 1), de taille (N,)
        y_scores (numpy vector): Vecteur des scores prédits, de taille (N,)

    Returns:
        precision (float): Précision
        recall (float): Rappel
    """
    threshold = compute_eer_threshold(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(np.int16)
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall


def compute_f1_score(y_true, y_scores):
    """
    Calcule le F1-score à partir des scores prédits
    Args:
        y_true (numpy vector): Vecteur des labels (0 ou 1), de taille (N,)
        y_scores (numpy vector): Vecteur des scores prédits, de taille (N,)
    Returns:
        f1_score (float): F1-score
    """
    precision, recall = compute_precision_recall(y_true, y_scores)
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score
