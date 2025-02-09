import logging
from src.models.random_forest_model import RandomForestModel
from src.models.svm_model import SVMModel
from src.models.neural_network_model import NeuralNetworkModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_evaluate_models(season_year, target_variable, use_pca=False):
    """Train and evaluate all models for a specific season and target.
    
    Args:
        season_year (int): Year of the season to use
        target_variable (str): Target variable to predict
        use_pca (bool): Whether to use PCA features
    """
    models = {
        'Random Forest': RandomForestModel(target_variable),
        'SVM': SVMModel(target_variable),
        'Neural Network': NeuralNetworkModel(target_variable)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name} model for {target_variable}...")
        
        # Load and prepare data
        model.load_data(season_year)
        model.prepare_data(use_pca)
        
        # Train model
        model.train()
        
        # Evaluate model
        f1, report = model.evaluate()
        results[name] = {
            'f1_score': f1,
            'report': report
        }
        
        # Get feature importance for Random Forest
        if name == 'Random Forest':
            importance = model.get_feature_importance()
            logger.info("\nFeature Importance:")
            logger.info(importance.head(10))
            
    return results

def main():
    """Main function to run model training and evaluation."""
    # Test configuration
    season = 2022
    target = 'has_pit_stop'
    use_pca = False
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Test Run - Season: {season}, Target: {target}, Use PCA: {use_pca}")
    logger.info('='*80)
    
    results = train_and_evaluate_models(season, target, use_pca)
    
    logger.info("\nSummary of Results:")
    for model_name, model_results in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"F1 Score: {model_results['f1_score']:.3f}")
        logger.info(f"Classification Report:\n{model_results['report']}")
    
    logger.info("\nTest run completed successfully. Ready to run full experiments.")

if __name__ == '__main__':
    main() 