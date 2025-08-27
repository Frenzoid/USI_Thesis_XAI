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
    1. Find and load evaluation result files with enhanced type detection
    2. Create individual experiment plots
    3. Generate comparison plots across multiple experiments
    4. Create comprehensive visualization reports
    5. Manage plot organization and file paths
    6. Auto-detect experiment types from names and file paths
    """
    
    def __init__(self):
        """Initialize plotting runner with visualization framework"""
        self.visualization_framework = VisualizationFramework()
        logger.info("PlottingRunner initialized")
    
    # =============================================================================
    # EXPERIMENT TYPE DETECTION
    # =============================================================================
    
    def extract_experiment_type_from_name(self, experiment_name: str) -> Optional[str]:
        """
        Extract experiment type from experiment name.
        
        Experiment names follow the pattern: {type}_{dataset}_{model}_{prompt}_{size}_{temp}
        For example: baseline_gmeg_gpt-4o-mini_gmeg_v1_basic_50_0p1
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            str: Experiment type if valid, None otherwise
        """
        if not experiment_name:
            return None
        
        # Extract the first part of the experiment name
        name_parts = experiment_name.split('_')
        if not name_parts:
            return None
        
        potential_type = name_parts[0]
        
        # Validate that it's a known experiment type
        if potential_type in Config.EXPERIMENT_TYPES:
            logger.debug(f"Extracted experiment type '{potential_type}' from name '{experiment_name}'")
            return potential_type
        
        return None
    
    def detect_experiment_type(self, experiment_name: str, experiment_file: str = None) -> str:
        """
        Detect experiment type using multiple methods with fallback chain.
        
        Detection priority:
        1. Extract from experiment name (most reliable)
        2. Extract from file path (fallback)
        3. Default to 'baseline' (ultimate fallback)
        
        Args:
            experiment_name: Name of the experiment
            experiment_file: Path to experiment file (optional)
            
        Returns:
            str: Detected experiment type
        """
        # Method 1: Extract from experiment name
        detected_type = self.extract_experiment_type_from_name(experiment_name)
        if detected_type:
            return detected_type
        
        # Method 2: Extract from file path
        if experiment_file:
            for exp_type in Config.EXPERIMENT_TYPES:
                if exp_type in experiment_file:
                    logger.debug(f"Detected experiment type '{exp_type}' from file path '{experiment_file}'")
                    return exp_type
        
        # Method 3: Ultimate fallback
        logger.warning(f"Could not detect experiment type for '{experiment_name}', using 'baseline' as fallback")
        return 'baseline'
    
    # =============================================================================
    # FILE DISCOVERY AND LOADING WITH ENHANCED TYPE DETECTION
    # =============================================================================
    
    def find_evaluation_files(self, experiment_type: Optional[str] = None) -> List[str]:
        """
        Find all evaluation result files for plotting.
        
        Args:
            experiment_type: Type filter for evaluation files
            
        Returns:
            list: Paths to evaluation files
            
        Raises:
            ValueError: If experiment type is invalid
        """
        if experiment_type:
            if not Config.validate_experiment_type(experiment_type):
                raise ValueError(f"Invalid experiment type: {experiment_type}")
            
            search_dir = Config.get_output_dirs_for_experiment_type(experiment_type)['evaluations']
            pattern = os.path.join(search_dir, "evaluation_*.json")
        else:
            # Search all experiment types
            pattern = os.path.join(Config.EVALUATIONS_DIR, "**", "evaluation_*.json")
        
        files = glob.glob(pattern, recursive=True)
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
    
    def load_evaluation_by_name(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load evaluation results for a specific experiment by name with enhanced type detection.
        
        Args:
            experiment_name: Name of experiment to load
            experiment_type: Optional type hint (will auto-detect if None)
            
        Returns:
            dict: Evaluation data, or None if not found
        """
        logger.info(f"Loading evaluation for: {experiment_name}")
        
        # Step 1: Determine experiment type using enhanced detection
        if not experiment_type:
            experiment_type = self.extract_experiment_type_from_name(experiment_name)
            if experiment_type:
                logger.info(f"Auto-detected experiment type: {experiment_type}")
        
        # Step 2: Find evaluation file using detected/provided type
        evaluation_file = None
        
        if experiment_type:
            # Search in the specific type directory first
            search_dir = Config.get_output_dirs_for_experiment_type(experiment_type)['evaluations']
            potential_file = os.path.join(search_dir, f"evaluation_{experiment_name}.json")
            if os.path.exists(potential_file):
                evaluation_file = potential_file
        
        # Step 3: Fallback to searching all directories if not found
        if not evaluation_file:
            logger.debug("Searching all evaluation directories as fallback")
            for exp_type in Config.EXPERIMENT_TYPES:
                search_dir = Config.get_output_dirs_for_experiment_type(exp_type)['evaluations']
                potential_file = os.path.join(search_dir, f"evaluation_{experiment_name}.json")
                if os.path.exists(potential_file):
                    evaluation_file = potential_file
                    logger.info(f"Found evaluation in {exp_type} directory")
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
        Format evaluation data for compatibility with visualization framework.
        
        Transforms evaluation results into the format expected by the visualization
        components, extracting key information and adding missing fields.
        
        Args:
            evaluation_data: Raw evaluation results
            
        Returns:
            dict: Formatted data ready for visualization
        """
        experiment_config = evaluation_data.get('original_experiment_config', {})
        
        # Create formatted result compatible with visualization framework
        formatted_result = {
            'batch_name': evaluation_data.get('original_experiment_name', 'unknown'),
            'experiment_name': evaluation_data.get('original_experiment_name', 'unknown'),
            'model_name': experiment_config.get('model', 'unknown'),
            'model_type': self._determine_model_type(experiment_config.get('model', '')),
            'prompt_key': experiment_config.get('prompt', 'unknown'),
            'dataset_type': evaluation_data.get('dataset_type', 'general'),
            'dataset_name': experiment_config.get('dataset', 'unknown'),
            'sample_size': evaluation_data.get('num_samples', 0),
            'num_valid_evaluations': evaluation_data.get('num_valid_evaluations', 0),
            'aggregated_scores': evaluation_data.get('aggregated_scores', {}),
            'timestamp': evaluation_data.get('evaluation_timestamp', datetime.now().isoformat())
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
        # Load models config to determine type
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
    # INDIVIDUAL EXPERIMENT PLOTTING WITH ENHANCED TYPE DETECTION
    # =============================================================================
    
    def create_individual_plot(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[str]:
        """
        Create individual plot for a single experiment with enhanced type detection.
        
        Args:
            experiment_name: Name of experiment to plot
            experiment_type: Optional type hint (will auto-detect if None)
            
        Returns:
            str: Path to generated plot file, or None if failed
        """
        logger.info(f"Creating individual plot for: {experiment_name}")
        
        # Load evaluation data using enhanced detection
        evaluation_data = self.load_evaluation_by_name(experiment_name, experiment_type)
        if not evaluation_data:
            logger.error(f"Could not load evaluation data for: {experiment_name}")
            return None
        
        # Format for visualization
        formatted_data = self.format_evaluation_for_visualization(evaluation_data)
        
        # Determine experiment type using enhanced detection
        final_experiment_type = self.detect_experiment_type(experiment_name)
        
        # Generate plot file path
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
    # COMPARISON PLOTTING WITH ENHANCED TYPE DETECTION
    # =============================================================================
    
    def create_comparison_plots(self, experiment_names: List[str], 
                              experiment_type: Optional[str] = None) -> Optional[List[str]]:
        """
        Create comparison plots for multiple experiments with enhanced type detection.
        
        Generates various comparison visualizations including metric comparisons
        and radar charts to analyze performance across experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            experiment_type: Optional type hint (will auto-detect for each experiment)
            
        Returns:
            list: Paths to generated plot files, or None if failed
        """
        logger.info(f"Creating comparison plots for {len(experiment_names)} experiments")
        
        # Load all evaluation data using enhanced detection
        evaluation_data_list = []
        detected_types = set()
        
        for experiment_name in experiment_names:
            # Use provided type or auto-detect for each experiment
            current_experiment_type = experiment_type
            if not current_experiment_type:
                current_experiment_type = self.extract_experiment_type_from_name(experiment_name)
            
            evaluation_data = self.load_evaluation_by_name(experiment_name, current_experiment_type)
            if evaluation_data:
                formatted_data = self.format_evaluation_for_visualization(evaluation_data)
                evaluation_data_list.append(formatted_data)
                
                # Track detected types for output directory decision
                detected_type = self.detect_experiment_type(experiment_name)
                detected_types.add(detected_type)
            else:
                logger.warning(f"Could not load data for: {experiment_name}")
        
        if not evaluation_data_list:
            logger.error("No valid evaluation data found for comparison")
            return None
        
        # Determine output directory based on detected types
        if experiment_type:
            final_experiment_type = experiment_type
        elif len(detected_types) == 1:
            final_experiment_type = detected_types.pop()
            logger.info(f"All experiments are of type: {final_experiment_type}")
        else:
            # Mixed types - use the most common or fallback to baseline
            type_counts = {}
            for exp_name in experiment_names:
                detected_type = self.extract_experiment_type_from_name(exp_name) or 'baseline'
                type_counts[detected_type] = type_counts.get(detected_type, 0) + 1
            
            final_experiment_type = max(type_counts, key=type_counts.get)
            logger.info(f"Mixed experiment types detected, using most common: {final_experiment_type}")
        
        # Generate output directory and comparison name
        output_dir = Config.get_output_dirs_for_experiment_type(final_experiment_type)['plots']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_name = f"comparison_{len(experiment_names)}exps_{timestamp}"
        
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
        
        Args:
            index_file: Path for the index file
            plot_files: List of plot file paths
            experiment_names: Names of compared experiments
            comparison_name: Name for this comparison
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
    # BATCH PLOTTING WITH ENHANCED TYPE DETECTION
    # =============================================================================
    
    def create_all_plots(self, experiment_type: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """
        Create plots for all evaluations of the specified type with enhanced detection.
        
        Generates individual plots for each evaluation and optionally creates
        a comprehensive comparison if multiple experiments are found.
        
        Args:
            experiment_type: Type filter for evaluations
            
        Returns:
            list: List of dictionaries with experiment names and plot file paths
        """
        logger.info(f"Creating plots for all evaluations (type: {experiment_type or 'all'})")
        
        # Find all evaluation files
        evaluation_files = self.find_evaluation_files(experiment_type)
        
        if not evaluation_files:
            logger.warning("No evaluation files found")
            return None
        
        results = []
        for file_path in evaluation_files:
            try:
                # Extract experiment name from file path
                file_name = os.path.basename(file_path)
                if file_name.startswith('evaluation_'):
                    experiment_name = file_name[11:-5]  # Remove 'evaluation_' prefix and '.json' suffix
                else:
                    experiment_name = file_name[:-5]  # Remove '.json' suffix
                
                # Use enhanced type detection
                detected_experiment_type = self.detect_experiment_type(experiment_name, file_path)
                
                # Create individual plot
                plot_file = self.create_individual_plot(experiment_name, detected_experiment_type)
                
                if plot_file:
                    results.append({
                        'experiment_name': experiment_name,
                        'experiment_type': detected_experiment_type,
                        'plot_file': plot_file
                    })
                
            except Exception as e:
                logger.error(f"Error creating plot for {file_path}: {e}")
                continue
        
        logger.info(f"Successfully created {len(results)} plots out of {len(evaluation_files)} evaluations")
        
        # Also create a comprehensive comparison if we have multiple results
        if len(results) > 1:
            try:
                experiment_names = [r['experiment_name'] for r in results]
                
                # Determine the most common experiment type for comparison
                if experiment_type:
                    comparison_type = experiment_type
                else:
                    # Use the most common detected type
                    type_counts = {}
                    for result in results:
                        result_type = result['experiment_type']
                        type_counts[result_type] = type_counts.get(result_type, 0) + 1
                    comparison_type = max(type_counts, key=type_counts.get) if type_counts else 'baseline'
                
                comparison_plots = self.create_comparison_plots(experiment_names, comparison_type)
                if comparison_plots:
                    logger.info(f"Also created comprehensive comparison with {len(comparison_plots)} files")
            except Exception as e:
                logger.warning(f"Could not create comprehensive comparison: {e}")
        
        return results