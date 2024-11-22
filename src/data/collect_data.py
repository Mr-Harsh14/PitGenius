from data_collector import F1DataCollector
import logging
import sys

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize the collector
    collector = F1DataCollector()
    
    try:
        logger.info("Starting data collection process...")
        
        # Collect data for 2024 Saudi Arabian GP Race
        logger.info("Initializing session for 2024 Saudi Arabian GP...")
        collector.initialize_session(2024, 'Saudi Arabia', 'R')
        
        logger.info("Starting data collection...")
        collector.collect_all_session_data()
        
        logger.info("Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 