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
    Handles generating plots and visualizations from evaluation results using metadata-based detection.
    
    This class serves as the interface between the CLI and the visualization
    framework, providing methods to:
    1. Find and load evaluation result files
    2. Extract metadata from file content (not filename parsing)
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
            dict: Extracted metadata including experiment_type, mode, model, etc.
        """
        # For evaluation files, metadata is in 'original_experiment_config'
        metadata = evaluation_data.get('original_experiment_config', {})
        
        # Extract key fields with defaults
        extracted = {
            'experiment_type': metadata.get('experiment_type', 'baseline'),
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
    # FILE DISCOVERY AND LOADING WITH METADATA-BASED DETECTION
    # =============================================================================
    
    def find_evaluation_files(self, experiment_type: Optional[str] = None) -> List[str]:
        """
        Find all evaluation result files for plotting.
        
        Args:
            experiment_type: Type filter for evaluation files (will be applied via metadata)
            
        Returns:
            list: Paths to evaluation files
        """
        # Search across all experiment types since we'll filter by metadata
        all_files = []
        for exp_type in Config.EXPERIMENT_TYPES:
            search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['evaluations']
            pattern = os.path.join(search_dir, "evaluation_*.json")
            files = glob.glob(pattern)
            all_files.extend(files)
        
        logger.info(f"Found {len(all_files)} evaluation files")
        return all_files
    
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
    
    def load_evaluation_by_name(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load evaluation results for a specific experiment by name with metadata-based detection.
        """
        logger.info(f"Loading evaluation for: {experiment_name}")
        
        # Search across all experiment types since we'll determine the actual type from metadata
        evaluation_file = None
        
        for exp_type in Config.EXPERIMENT_TYPES:
            search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['evaluations']
            potential_file = os.path.join(search_dir, f"evaluation_{experiment_name}.json")
            if os.path.exists(potential_file):
                evaluation_file = potential_file
                break
        
        if not evaluation_file:
            logger.error(f"Evaluation file not found: {experiment_name}")
            return None
        
        return self.load_evaluation_results(evaluation_file)
    
    # =============================================================================
    # DATA FORMATTING FOR VISUALIZATION
    # =============================================================================
    
    def format_evaluation_for_visualization(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format evaluation data for compatibility with visualization framework using metadata-based detection.
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
            'experiment_type': metadata['experiment_type'],
            'dataset_type': evaluation_data.get('dataset_type', 'general'),
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
    # INDIVIDUAL EXPERIMENT PLOTTING WITH METADATA-BASED DETECTION
    # =============================================================================
    
    def create_individual_plot(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[str]:
        """
        Create individual plot for a single experiment with metadata-based detection.
        """
        logger.info(f"Creating individual plot for: {experiment_name}")
        
        # Load evaluation data
        evaluation_data = self.load_evaluation_by_name(experiment_name, experiment_type)
        if not evaluation_data:
            logger.error(f"Could not load evaluation data for: {experiment_name}")
            return None
        
        # Format for visualization using metadata-based approach
        formatted_data = self.format_evaluation_for_visualization(evaluation_data)
        
        # Extract actual experiment type from metadata
        metadata = self.extract_metadata_from_evaluation_data(evaluation_data)
        final_experiment_type = metadata['experiment_type']
        
        # Generate plot file path using the correct type from metadata
        file_paths = Config.generate_file_paths(final_experiment_type, experiment_name)
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
    # COMPARISON PLOTTING WITH METADATA-BASED DETECTION
    # =============================================================================
    
    def create_comparison_plots(self, experiment_names: List[str], 
                              experiment_type: Optional[str] = None) -> Optional[List[str]]:
        """
        Create comparison plots for multiple experiments with metadata-based detection.
        """
        logger.info(f"Creating comparison plots for {len(experiment_names)} experiments")
        
        # Load all evaluation data using metadata-based detection
        evaluation_data_list = []
        detected_types = set()
        
        for experiment_name in experiment_names:
            evaluation_data = self.load_evaluation_by_name(experiment_name, experiment_type)
            if evaluation_data:
                # Apply experiment type filter if specified
                if experiment_type:
                    metadata = self.extract_metadata_from_evaluation_data(evaluation_data)
                    file_experiment_type = metadata['experiment_type']
                    if file_experiment_type != experiment_type:
                        logger.info(f"Skipping {experiment_name}: type {file_experiment_type} doesn't match filter {experiment_type}")
                        continue
                
                formatted_data = self.format_evaluation_for_visualization(evaluation_data)
                evaluation_data_list.append(formatted_data)
                
                # Track detected types for output directory decision
                metadata = self.extract_metadata_from_evaluation_data(evaluation_data)
                detected_types.add(metadata['experiment_type'])
            else:
                logger.warning(f"Could not load data for: {experiment_name}")
        
        if not evaluation_data_list:
            logger.error("No valid evaluation data found for comparison")
            return None
        
        # Determine output directory based on detected types or filter
        if experiment_type:
            final_experiment_type = experiment_type
        elif len(detected_types) == 1:
            final_experiment_type = detected_types.pop()
            logger.info(f"All experiments are of type: {final_experiment_type}")
        else:
            # Mixed types - use the most common
            type_counts = {}
            for data in evaluation_data_list:
                exp_type = data['experiment_type']
                type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
            
            final_experiment_type = max(type_counts, key=lambda k: type_counts[k])
            logger.info(f"Mixed experiment types detected, using most common: {final_experiment_type}")
        
        # Generate output directory and comparison name
        output_dir = Config.get_output_dirs_for_experiment_type(final_experiment_type)['plots']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_name = f"comparison_{len(evaluation_data_list)}exps_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        plot_files = []
        
        try:
            # Create different types of comparison plots
            
            # 1. F1 Score comparison
            f1_plot_file = os.path.join(output_dir, f"{comparison_name}_f1_comparison.html")
            f1_fig = self.visualization_framework.plot_metric_comparison(
                evaluation_data_list, metric='f1_score', save_path=f1_plot_file
            )
            plot_files.append(f1_plot_file)
            
            # 2. Semantic similarity comparison
            similarity_plot_file = os.path.join(output_dir, f"{comparison_name}_similarity_comparison.html")
            similarity_fig = self.visualization_framework.plot_metric_comparison(
                evaluation_data_list, metric='semantic_similarity', save_path=similarity_plot_file
            )
            plot_files.append(similarity_plot_file)
            
            # 3. Radar chart comparison (if multiple experiments)
            if len(evaluation_data_list) > 1:
                radar_plot_file = os.path.join(output_dir, f"{comparison_name}_radar_comparison.html")
                radar_fig = self.visualization_framework.plot_model_comparison_radar(
                    evaluation_data_list, save_path=radar_plot_file
                )
                plot_files.append(radar_plot_file)
            
            # 4. Create index file for easy navigation
            index_file = os.path.join(output_dir, f"{comparison_name}_index.html")
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
    # BATCH PLOTTING WITH METADATA-BASED DETECTION
    # =============================================================================
    
    def create_all_plots(self, experiment_type: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """
        Create plots for all evaluations with metadata-based filtering.
        """
        logger.info(f"Creating plots for all evaluations (type filter: {experiment_type or 'none'})")
        
        # Find evaluation files across all types since we'll filter by metadata
        all_evaluation_files = []
        for exp_type in Config.EXPERIMENT_TYPES:
            search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['evaluations']
            pattern = os.path.join(search_dir, "evaluation_*.json")
            files = glob.glob(pattern)
            all_evaluation_files.extend(files)
        
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
                file_experiment_type = metadata['experiment_type']
                
                # Apply experiment type filter if specified
                if experiment_type and file_experiment_type != experiment_type:
                    logger.debug(f"Skipping {experiment_name}: type {file_experiment_type} doesn't match filter {experiment_type}")
                    continue
                
                # Create individual plot
                plot_file = self.create_individual_plot(experiment_name, file_experiment_type)
                
                if plot_file:
                    results.append({
                        'experiment_name': experiment_name,
                        'experiment_type': file_experiment_type,
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
                
                # Determine the most common experiment type for comparison
                if experiment_type:
                    comparison_type = experiment_type
                else:
                    type_counts = {}
                    for result in results:
                        result_type = result['experiment_type']
                        type_counts[result_type] = type_counts.get(result_type, 0) + 1
                    comparison_type = max(type_counts, key=lambda k: type_counts[k]) if type_counts else 'baseline'
                
                comparison_plots = self.create_comparison_plots(experiment_names, comparison_type)
                if comparison_plots:
                    logger.info(f"Also created comprehensive comparison with {len(comparison_plots)} files")
            except Exception as e:
                logger.warning(f"Could not create comprehensive comparison: {e}")
        
        return results