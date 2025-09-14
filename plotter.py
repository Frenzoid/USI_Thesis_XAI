import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional

from config import Config
from utils import setup_logging
from visualization import VisualizationFramework

logger = setup_logging("plotting_runner")

class PlottingRunner:
    """
    Handles generating plots and visualizations from evaluation results.
    
    This class serves as the interface between the CLI and the visualization
    framework, providing methods to:
    1. Find and load evaluation result files
    2. Extract metadata from file content
    3. Create individual experiment plots
    4. Generate comparison plots across multiple experiments
    5. Create comprehensive visualization reports
    6. Manage plot organization and file paths
    """
    
    def __init__(self):
        """Initialize plotting runner with visualization framework"""
        self.visualization_framework = VisualizationFramework()
        logger.info("PlottingRunner initialized")
    
    # =============================================================================
    # METADATA EXTRACTION FROM FILE CONTENT
    # =============================================================================
    
    def extract_metadata_from_evaluation_data(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract experiment metadata from loaded evaluation file data.
        
        Args:
            evaluation_data: Loaded evaluation result data
            
        Returns:
            dict: Extracted metadata including mode, model, etc.
        """
        # For evaluation files, metadata is in 'original_experiment_config'
        metadata = evaluation_data.get('original_experiment_config', {})
        
        # Extract key fields with defaults
        extracted = {
            'mode': metadata.get('mode', 'zero-shot'),
            'model': metadata.get('model', 'unknown'),
            'dataset': metadata.get('dataset', 'unknown'),
            'prompt': metadata.get('prompt', 'unknown'),
            'size': metadata.get('size', 0),
            'temperature': metadata.get('temperature', 0.1),
            'few_shot_row': metadata.get('few_shot_row')
        }
        
        logger.debug(f"Extracted metadata from evaluation file: {extracted}")
        return extracted
    
    # =============================================================================
    # FILE DISCOVERY AND LOADING
    # =============================================================================
    
    def find_evaluation_files(self) -> List[str]:
        """
        Find all evaluation result files for plotting.
        
        Returns:
            list: Paths to evaluation files
        """
        pattern = os.path.join(Config.EVALUATIONS_DIR, "evaluation_*.json")
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} evaluation files")
        return files
    
    def load_evaluation_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load evaluation results from JSON file.
        
        Args:
            file_path: Path to evaluation result file
            
        Returns:
            dict: Loaded evaluation data, or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded evaluation results from: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading evaluation results from {file_path}: {e}")
            return None
    
    def load_evaluation_by_name(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        Load evaluation results for a specific experiment by name.
        """
        logger.info(f"Loading evaluation for: {experiment_name}")
        
        evaluation_file = os.path.join(Config.EVALUATIONS_DIR, f"evaluation_{experiment_name}.json")
        
        if not os.path.exists(evaluation_file):
            logger.error(f"Evaluation file not found: {experiment_name}")
            return None
        
        return self.load_evaluation_results(evaluation_file)
    
    # =============================================================================
    # DATA FORMATTING FOR VISUALIZATION
    # =============================================================================
    
    def format_evaluation_for_visualization(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format evaluation data for compatibility with visualization framework.
        """
        # Extract metadata from file content
        metadata = self.extract_metadata_from_evaluation_data(evaluation_data)
        
        # Create formatted result compatible with visualization framework
        formatted_result = {
            'batch_name': evaluation_data.get('original_experiment_name', 'unknown'),
            'experiment_name': evaluation_data.get('original_experiment_name', 'unknown'),
            'model_name': metadata['model'],
            'model_type': self._determine_model_type(metadata['model']),
            'prompt_key': metadata['prompt'],
            'mode': metadata['mode'],
            'dataset_type': 'general',
            'dataset_name': metadata['dataset'],
            'sample_size': metadata['size'],
            'temperature': metadata['temperature'],
            'few_shot_row': metadata['few_shot_row'],
            'num_valid_evaluations': evaluation_data.get('num_valid_evaluations', 0),
            'aggregated_scores': evaluation_data.get('aggregated_scores', {}),
            'timestamp': evaluation_data.get('evaluation_timestamp', datetime.now().isoformat()),
            'extracted_metadata': metadata
        }
        
        return formatted_result
    
    def _determine_model_type(self, model_name: str) -> str:
        """
        Determine model type from model name for visualization coloring.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Model type ('api' or 'local')
        """
        try:
            models_config = Config.load_models_config()
            if model_name in models_config:
                return models_config[model_name]['type']
        except:
            pass
        
        # Fallback logic based on common naming patterns
        if any(api_indicator in model_name.lower() for api_indicator in ['gpt', 'gemini', 'claude']):
            return 'api'
        else:
            return 'local'
    
    # =============================================================================
    # INDIVIDUAL EXPERIMENT PLOTTING
    # =============================================================================
    
    def create_individual_plot(self, experiment_name: str) -> Optional[str]:
        """
        Create individual plot for a single experiment.
        """
        logger.info(f"Creating individual plot for: {experiment_name}")
        
        # Load evaluation data
        evaluation_data = self.load_evaluation_by_name(experiment_name)
        if not evaluation_data:
            logger.error(f"Could not load evaluation data for: {experiment_name}")
            return None
        
        # Format for visualization
        formatted_data = self.format_evaluation_for_visualization(evaluation_data)
        
        # Generate plot file path
        file_paths = Config.generate_file_paths(experiment_name)
        plot_file = file_paths['plot']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        
        try:
            # Create individual metrics plot
            fig = self.visualization_framework.create_individual_experiment_plot(formatted_data)
            
            # Save plot as interactive HTML
            fig.write_html(plot_file)
            
            logger.info(f"Individual plot created: {plot_file}")
            return plot_file
            
        except Exception as e:
            logger.error(f"Error creating individual plot for {experiment_name}: {e}")
            return None
    
    # =============================================================================
    # COMPARISON PLOTTING
    # =============================================================================
    
    def create_comparison_plots(self, experiment_names: List[str]) -> Optional[List[str]]:
        """
        Create comparison plots for multiple experiments.
        """
        logger.info(f"Creating comparison plots for {len(experiment_names)} experiments")
        
        # Load all evaluation data
        evaluation_data_list = []
        
        for experiment_name in experiment_names:
            evaluation_data = self.load_evaluation_by_name(experiment_name)
            if evaluation_data:
                formatted_data = self.format_evaluation_for_visualization(evaluation_data)
                evaluation_data_list.append(formatted_data)
            else:
                logger.warning(f"Could not load data for: {experiment_name}")
        
        if not evaluation_data_list:
            logger.error("No valid evaluation data found for comparison")
            return None
        
        # Generate output directory and comparison name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_name = f"comparison_{len(evaluation_data_list)}exps_{timestamp}"
        
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
        
        plot_files = []
        
        try:
            # Create different types of comparison plots
            
            # 1. F1 Score comparison
            f1_plot_file = os.path.join(Config.PLOTS_DIR, f"{comparison_name}_f1_comparison.html")
            f1_fig = self.visualization_framework.plot_metric_comparison(
                evaluation_data_list, metric='f1_score', save_path=f1_plot_file
            )
            plot_files.append(f1_plot_file)
            
            # 2. Semantic similarity comparison
            similarity_plot_file = os.path.join(Config.PLOTS_DIR, f"{comparison_name}_similarity_comparison.html")
            similarity_fig = self.visualization_framework.plot_metric_comparison(
                evaluation_data_list, metric='semantic_similarity', save_path=similarity_plot_file
            )
            plot_files.append(similarity_plot_file)
            
            # 3. Radar chart comparison (if multiple experiments)
            if len(evaluation_data_list) > 1:
                radar_plot_file = os.path.join(Config.PLOTS_DIR, f"{comparison_name}_radar_comparison.html")
                radar_fig = self.visualization_framework.plot_model_comparison_radar(
                    evaluation_data_list, save_path=radar_plot_file
                )
                plot_files.append(radar_plot_file)
            
            # 4. Create index file for easy navigation
            index_file = os.path.join(Config.PLOTS_DIR, f"{comparison_name}_index.html")
            self._create_comparison_index(index_file, plot_files, experiment_names, comparison_name)
            plot_files.append(index_file)
            
            logger.info(f"Comparison plots created: {len(plot_files)} files")
            return plot_files
            
        except Exception as e:
            logger.error(f"Error creating comparison plots: {e}")
            return None
    
    def _create_comparison_index(self, index_file: str, plot_files: List[str], 
                               experiment_names: List[str], comparison_name: str):
        """
        Create an HTML index file for comparison plots.
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Comparison: {comparison_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .plot-link {{ display: inline-block; margin: 10px; padding: 10px 20px; 
                             background: #007bff; color: white; text-decoration: none; 
                             border-radius: 5px; }}
                .plot-link:hover {{ background: #0056b3; }}
                .experiment-list {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Experiment Comparison Report</h1>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Comparison Name:</strong> {comparison_name}</p>
                <p><strong>Number of Experiments:</strong> {len(experiment_names)}</p>
                <p><strong>Number of Plots:</strong> {len(plot_files)}</p>
            </div>
            
            <div class="experiment-list">
                <h2>Experiments Compared</h2>
                <ul>
        """
        
        for exp_name in experiment_names:
            html_content += f"<li>{exp_name}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <h2>Generated Plots</h2>
            <div>
        """
        
        plot_names = {
            'f1_comparison.html': 'F1 Score Comparison',
            'similarity_comparison.html': 'Semantic Similarity Comparison',
            'radar_comparison.html': 'Radar Chart Comparison'
        }
        
        for plot_file in plot_files:
            if plot_file == index_file:  # Skip the index file itself
                continue
            
            file_name = os.path.basename(plot_file)
            
            # Determine display name
            display_name = file_name
            for key, name in plot_names.items():
                if key in file_name:
                    display_name = name
                    break
            
            html_content += f'<a href="{file_name}" class="plot-link">{display_name}</a>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(index_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Comparison index created: {index_file}")
    
    # =============================================================================
    # BATCH PLOTTING
    # =============================================================================
    
    def create_all_plots(self) -> Optional[List[Dict[str, str]]]:
        """
        Create plots for all evaluations.
        """
        logger.info("Creating plots for all evaluations")
        
        # Find evaluation files
        all_evaluation_files = self.find_evaluation_files()
        
        if not all_evaluation_files:
            logger.warning("No evaluation files found")
            return None
        
        results = []
        for file_path in all_evaluation_files:
            try:
                # Extract experiment name from file path
                file_name = os.path.basename(file_path)
                if file_name.startswith('evaluation_'):
                    experiment_name = file_name[11:-5]  # Remove 'evaluation_' prefix and '.json' suffix
                else:
                    experiment_name = file_name[:-5]  # Remove '.json' suffix
                
                # Load evaluation data
                evaluation_data = self.load_evaluation_results(file_path)
                if not evaluation_data:
                    continue
                
                # Extract metadata from file content
                metadata = self.extract_metadata_from_evaluation_data(evaluation_data)
                
                # Create individual plot
                plot_file = self.create_individual_plot(experiment_name)
                
                if plot_file:
                    results.append({
                        'experiment_name': experiment_name,
                        'mode': metadata['mode'],
                        'model': metadata['model'],
                        'plot_file': plot_file
                    })
                
            except Exception as e:
                logger.error(f"Error creating plot for {file_path}: {e}")
                continue
        
        logger.info(f"Successfully created {len(results)} plots")
        
        # Create comprehensive comparison if we have multiple results
        if len(results) > 1:
            try:
                experiment_names = [r['experiment_name'] for r in results]
                comparison_plots = self.create_comparison_plots(experiment_names)
                if comparison_plots:
                    logger.info(f"Also created comprehensive comparison with {len(comparison_plots)} files")
            except Exception as e:
                logger.warning(f"Could not create comprehensive comparison: {e}")
        
        return results