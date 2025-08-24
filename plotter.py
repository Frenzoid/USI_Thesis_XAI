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
    """Handles generating plots and visualizations from evaluation results"""
    
    def __init__(self):
        self.visualization_framework = VisualizationFramework()
        
        # Ensure visualization framework has the individual plot method
        if not hasattr(self.visualization_framework, 'create_individual_experiment_plot'):
            # Add the method if it doesn't exist
            import plotly.graph_objects as go
            
            def create_individual_experiment_plot(self, evaluation_result):
                """Create a plot for an individual experiment"""
                experiment_name = evaluation_result.get('batch_name', 'Unknown Experiment')
                agg_scores = evaluation_result.get('aggregated_scores', {})
                
                if not agg_scores:
                    # Create empty plot
                    fig = go.Figure()
                    fig.add_annotation(text="No evaluation metrics available", 
                                      xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                    fig.update_layout(title=f"Results for {experiment_name}")
                    return fig
                
                # Extract metrics for plotting
                metrics = []
                values = []
                errors = []
                
                for metric, stats in agg_scores.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        metrics.append(metric.replace('_', ' ').title())
                        values.append(stats['mean'])
                        errors.append(stats.get('std', 0))
                
                if not metrics:
                    # Create empty plot
                    fig = go.Figure()
                    fig.add_annotation(text="No valid metrics found", 
                                      xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                    fig.update_layout(title=f"Results for {experiment_name}")
                    return fig
                
                # Create bar plot with error bars
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=metrics,
                    y=values,
                    error_y=dict(type='data', array=errors),
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title=f'Evaluation Results: {experiment_name}',
                    xaxis_title='Metrics',
                    yaxis_title='Score',
                    yaxis=dict(range=[0, 1]),
                    height=500,
                    showlegend=False
                )
                
                return fig
            
            # Bind the method to the instance
            import types
            self.visualization_framework.create_individual_experiment_plot = types.MethodType(
                create_individual_experiment_plot, self.visualization_framework
            )
        
        logger.info("PlottingRunner initialized")
    
    def find_evaluation_files(self, experiment_type: Optional[str] = None) -> List[str]:
        """Find all evaluation result files"""
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
        """Load evaluation results from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded evaluation results from: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading evaluation results from {file_path}: {e}")
            return None
    
    def load_evaluation_by_name(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load evaluation results for a specific experiment by name"""
        logger.info(f"Loading evaluation for: {experiment_name}")
        
        # Find the evaluation file
        if experiment_type:
            search_dirs = [Config.get_output_dirs_for_experiment_type(experiment_type)['evaluations']]
        else:
            # Search all experiment types
            search_dirs = []
            for exp_type in Config.EXPERIMENT_TYPES:
                search_dirs.append(Config.get_output_dirs_for_experiment_type(exp_type)['evaluations'])
        
        evaluation_file = None
        for search_dir in search_dirs:
            potential_file = os.path.join(search_dir, f"evaluation_{experiment_name}.json")
            if os.path.exists(potential_file):
                evaluation_file = potential_file
                break
        
        if not evaluation_file:
            logger.error(f"Evaluation file not found: {experiment_name}")
            return None
        
        return self.load_evaluation_results(evaluation_file)
    
    def format_evaluation_for_visualization(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format evaluation data for visualization framework"""
        experiment_config = evaluation_data.get('original_experiment_config', {})
        
        # Create formatted result compatible with existing visualization framework
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
        """Determine model type from model name"""
        # Load models config to determine type
        try:
            models_config = Config.load_models_config()
            if model_name in models_config:
                return models_config[model_name]['type']
        except:
            pass
        
        # Fallback logic
        if any(api_indicator in model_name.lower() for api_indicator in ['gpt', 'gemini', 'claude']):
            return 'api'
        else:
            return 'local'
    
    def create_individual_plot(self, experiment_name: str, experiment_type: Optional[str] = None) -> Optional[str]:
        """Create individual plot for a single experiment"""
        logger.info(f"Creating individual plot for: {experiment_name}")
        
        # Load evaluation data
        evaluation_data = self.load_evaluation_by_name(experiment_name, experiment_type)
        if not evaluation_data:
            logger.error(f"Could not load evaluation data for: {experiment_name}")
            return None
        
        # Format for visualization
        formatted_data = self.format_evaluation_for_visualization(evaluation_data)
        
        # Determine experiment type if not provided
        if not experiment_type:
            for exp_type in Config.EXPERIMENT_TYPES:
                # Check if experiment name contains experiment type
                if exp_type in experiment_name:
                    experiment_type = exp_type
                    break
            else:
                experiment_type = 'baseline'  # Default
        
        # Generate plot file path
        file_paths = Config.generate_file_paths(experiment_type, experiment_name)
        plot_file = file_paths['plot']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        
        try:
            # Create individual metrics plot
            fig = self.visualization_framework.create_individual_experiment_plot(formatted_data)
            
            # Save plot
            fig.write_html(plot_file)
            
            logger.info(f"Individual plot created: {plot_file}")
            return plot_file
            
        except Exception as e:
            logger.error(f"Error creating individual plot for {experiment_name}: {e}")
            return None
    
    def create_comparison_plots(self, experiment_names: List[str], 
                              experiment_type: Optional[str] = None) -> Optional[List[str]]:
        """Create comparison plots for multiple experiments"""
        logger.info(f"Creating comparison plots for {len(experiment_names)} experiments")
        
        # Load all evaluation data
        evaluation_data_list = []
        for experiment_name in experiment_names:
            evaluation_data = self.load_evaluation_by_name(experiment_name, experiment_type)
            if evaluation_data:
                formatted_data = self.format_evaluation_for_visualization(evaluation_data)
                evaluation_data_list.append(formatted_data)
            else:
                logger.warning(f"Could not load data for: {experiment_name}")
        
        if not evaluation_data_list:
            logger.error("No valid evaluation data found for comparison")
            return None
        
        # Determine experiment type if not provided
        if not experiment_type:
            experiment_type = 'baseline'  # Default
            for exp_name in experiment_names:
                for exp_type in Config.EXPERIMENT_TYPES:
                    if exp_type in exp_name:
                        experiment_type = exp_type
                        break
        
        # Generate output directory
        output_dir = Config.get_output_dirs_for_experiment_type(experiment_type)['plots']
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
    
    def create_all_plots(self, experiment_type: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """Create plots for all evaluations of the specified type"""
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
                
                # Create individual plot
                plot_file = self.create_individual_plot(experiment_name, experiment_type)
                
                if plot_file:
                    results.append({
                        'experiment_name': experiment_name,
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
                comparison_plots = self.create_comparison_plots(experiment_names, experiment_type)
                if comparison_plots:
                    logger.info(f"Also created comprehensive comparison with {len(comparison_plots)} files")
            except Exception as e:
                logger.warning(f"Could not create comprehensive comparison: {e}")
        
        return results
    
    def _create_comparison_index(self, index_file: str, plot_files: List[str], 
                               experiment_names: List[str], comparison_name: str):
        """Create an HTML index file for comparison plots"""
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