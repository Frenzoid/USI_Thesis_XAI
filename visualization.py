import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Dict, Any

from config import Config
from utils import setup_logging

logger = setup_logging("visualization")

class VisualizationFramework:
    """
    Comprehensive visualization system for experiment results.
    
    This framework provides:
    1. Individual experiment visualizations
    2. Comparative analysis across experiments
    3. Model performance comparisons
    4. Prompt engineering analysis
    5. Interactive HTML reports with navigation
    6. Automated plot generation and organization
    """
    
    def __init__(self):
        """Initialize visualization framework and create directory structure"""
        # Base directory for all plots
        self.plots_base_dir = Config.PLOTS_DIR
        self.ensure_directory_structure()
        
        logger.info("VisualizationFramework initialized")
    
    # =============================================================================
    # DIRECTORY MANAGEMENT
    # =============================================================================
    
    def ensure_directory_structure(self):
        """
        Ensure proper directory structure exists for organized plot storage.
        
        Creates subdirectories for different types of visualizations
        to keep outputs organized and easy to navigate.
        """
        directories = [
            self.plots_base_dir,
            os.path.join(self.plots_base_dir, "individual_experiments"),
            os.path.join(self.plots_base_dir, "comparative_analysis"),
            os.path.join(self.plots_base_dir, "comprehensive_reports"),
            os.path.join(self.plots_base_dir, "model_comparisons"),
            os.path.join(self.plots_base_dir, "prompt_comparisons")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.debug(f"Ensured {len(directories)} plot directories exist")
    
    # =============================================================================
    # METRIC COMPARISON VISUALIZATIONS
    # =============================================================================
    
    def plot_metric_comparison(self, evaluations: List[Dict], metric: str = 'f1_score',
                              save_path: str = None, show_error_bars: bool = True) -> go.Figure:
        """
        Create comparative bar chart for a specific metric across evaluations.
        
        Features modern styling with color-coded bars, error bars, and average lines.
        
        Args:
            evaluations: List of evaluation result dictionaries
            metric: Metric name to plot (e.g., 'f1_score', 'semantic_similarity')
            save_path: Optional path to save the plot
            show_error_bars: Whether to show standard deviation error bars
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot figure
        """
        logger.info(f"Creating metric comparison plot for '{metric}' across {len(evaluations)} evaluations")
        
        names = []
        means = []
        stds = []
        colors = []
        
        # Color scheme based on model type for easy identification
        color_map = {
            'open_source': '#1f77b4',  # Blue for open source models
            'api': '#ff7f0e',          # Orange for API models
            'gpt': '#2ca02c',          # Green for GPT models
            'gemini': '#d62728',       # Red for Gemini models
            'claude': '#9467bd'        # Purple for Claude models
        }
        
        # Process each evaluation result
        for eval_result in evaluations:
            name = eval_result.get('batch_name', 'Unknown')
            model_type = eval_result.get('model_type', 'unknown')
            model_name = eval_result.get('model_name', '')
            
            names.append(name)
            agg_scores = eval_result['aggregated_scores']
            
            # Extract metric values with fallback to 0
            if metric in agg_scores:
                means.append(agg_scores[metric]['mean'])
                stds.append(agg_scores[metric]['std'] if show_error_bars else 0)
            else:
                means.append(0)
                stds.append(0)
            
            # Determine color based on model characteristics
            if 'gpt' in model_name.lower():
                colors.append(color_map.get('gpt', color_map['api']))
            elif 'gemini' in model_name.lower():
                colors.append(color_map.get('gemini', color_map['api']))
            elif 'claude' in model_name.lower():
                colors.append(color_map.get('claude', color_map['api']))
            else:
                colors.append(color_map.get(model_type, '#666666'))
        
        # Create interactive bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=names,
            y=means,
            error_y=dict(type='data', array=stds) if show_error_bars else None,
            name=metric.replace('_', ' ').title(),
            marker_color=colors,
            text=[f'{m:.3f}' for m in means],
            textposition='auto'
        ))
        
        # Style the plot with modern design
        fig.update_layout(
            title=f'{metric.replace("_", " ").title()} Comparison Across Models',
            xaxis_title='Experiment',
            yaxis_title=metric.replace('_', ' ').title(),
            showlegend=False,
            xaxis_tickangle=-45,
            height=600,
            margin=dict(b=150),  # Extra space for rotated labels
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black")
        )
        
        # Add horizontal line showing average performance
        if means:
            avg_score = np.mean(means)
            fig.add_hline(y=avg_score, line_dash="dash", line_color="gray",
                         annotation_text=f"Average: {avg_score:.3f}")
        
        # Save plot if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved {metric} comparison plot to: {save_path}")
        
        return fig
    
    # =============================================================================
    # RADAR CHART VISUALIZATIONS
    # =============================================================================
    
    def plot_model_comparison_radar(self, evaluations: List[Dict], 
                                   metrics: List[str] = None, save_path: str = None,
                                   max_models: int = 8) -> go.Figure:
        """
        Create radar chart comparing multiple models across various metrics.
        
        Radar charts are excellent for comparing multi-dimensional performance
        and identifying models that excel in specific areas.
        
        Args:
            evaluations: List of evaluation results
            metrics: List of metrics to include in radar (None for default set)
            save_path: Optional path to save the plot
            max_models: Maximum number of models to show (for readability)
            
        Returns:
            plotly.graph_objects.Figure: Interactive radar chart
        """
        logger.info(f"Creating radar chart for {len(evaluations)} evaluations")
        
        if metrics is None:
            metrics = ['f1_score', 'semantic_similarity', 'precision', 'recall', 'exact_match']
        
        fig = go.Figure()
        
        # Limit number of models for chart readability
        evaluations_to_plot = evaluations[:max_models] if len(evaluations) > max_models else evaluations
        
        if len(evaluations) > max_models:
            logger.warning(f"Limiting radar chart to {max_models} models for readability")
        
        # Use distinct colors for each model
        colors = px.colors.qualitative.Set3[:len(evaluations_to_plot)]
        
        for i, eval_result in enumerate(evaluations_to_plot):
            name = eval_result.get('batch_name', 'Unknown')
            model_name = eval_result.get('model_name', '')
            prompt_key = eval_result.get('prompt_key', '')
            
            # Create shorter, more readable label
            short_name = f"{model_name}-{prompt_key}" if len(name) > 20 else name
            
            agg_scores = eval_result['aggregated_scores']
            
            values = []
            labels = []
            
            # Extract metric values
            for metric in metrics:
                if metric in agg_scores:
                    values.append(agg_scores[metric]['mean'])
                    labels.append(metric.replace('_', ' ').title())
            
            # Close the radar chart by repeating first value
            if values:
                values.append(values[0])
                labels.append(labels[0])
            
            # Add trace for this model
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=short_name,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.6
            ))
        
        # Style the radar chart
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            title=dict(
                text='Model Performance Comparison (Radar Chart)',
                font=dict(size=16, family="Arial Black")
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05
            ),
            height=600,
            width=800
        )
        
        # Save plot if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved radar chart to: {save_path}")
        
        return fig
    
    # =============================================================================
    # PROMPT COMPARISON VISUALIZATIONS
    # =============================================================================
    
    def plot_prompt_comparison(self, evaluations: List[Dict], group_by_model: bool = True,
                              save_path: str = None) -> go.Figure:
        """
        Create visualization comparing prompt engineering effects.
        
        Can group by model (showing prompt effects per model) or by prompt
        (showing model performance per prompt type).
        
        Args:
            evaluations: List of evaluation results
            group_by_model: If True, group by model; if False, group by prompt
            save_path: Optional path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Comparison plot
        """
        logger.info(f"Creating prompt comparison plot, group_by_model={group_by_model}")
        
        if group_by_model:
            # Group by model, compare prompts within each model
            model_groups = {}
            for eval_result in evaluations:
                model_name = eval_result.get('model_name', 'Unknown')
                if model_name not in model_groups:
                    model_groups[model_name] = []
                model_groups[model_name].append(eval_result)
            
            # Create subplots for each model
            fig = make_subplots(
                rows=1, cols=len(model_groups),
                subplot_titles=list(model_groups.keys()),
                shared_yaxes=True
            )
            
            for col, (model_name, model_evals) in enumerate(model_groups.items(), 1):
                prompt_names = [eval_result.get('prompt_key', 'Unknown') for eval_result in model_evals]
                f1_scores = []
                
                for eval_result in model_evals:
                    agg_scores = eval_result.get('aggregated_scores', {})
                    f1_scores.append(agg_scores.get('f1_score', {}).get('mean', 0))
                
                fig.add_trace(
                    go.Bar(x=prompt_names, y=f1_scores, name=model_name, showlegend=(col == 1)),
                    row=1, col=col
                )
            
            fig.update_layout(
                title="Prompt Engineering Effects by Model",
                height=500,
                font=dict(size=10)
            )
            
        else:
            # Group by prompt, compare models within each prompt
            prompt_groups = {}
            for eval_result in evaluations:
                prompt_key = eval_result.get('prompt_key', 'Unknown')
                if prompt_key not in prompt_groups:
                    prompt_groups[prompt_key] = []
                prompt_groups[prompt_key].append(eval_result)
            
            fig = go.Figure()
            
            for prompt_key, prompt_evals in prompt_groups.items():
                model_names = [eval_result.get('model_name', 'Unknown') for eval_result in prompt_evals]
                f1_scores = []
                
                for eval_result in prompt_evals:
                    agg_scores = eval_result.get('aggregated_scores', {})
                    f1_scores.append(agg_scores.get('f1_score', {}).get('mean', 0))
                
                fig.add_trace(go.Bar(x=model_names, y=f1_scores, name=prompt_key))
            
            fig.update_layout(
                title="Model Performance by Prompt Type",
                xaxis_title="Model",
                yaxis_title="F1 Score",
                height=500,
                barmode='group'
            )
        
        # Save plot if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved prompt comparison plot to: {save_path}")
        
        return fig
    
    # =============================================================================
    # INDIVIDUAL EXPERIMENT VISUALIZATION
    # =============================================================================
    
    def create_individual_experiment_plot(self, evaluation_result):
        """
        Create a comprehensive plot for an individual experiment.
        
        Shows all metrics as a bar chart with error bars and clear labeling.
        
        Args:
            evaluation_result: Single evaluation result dictionary
            
        Returns:
            plotly.graph_objects.Figure: Individual experiment plot
        """
        experiment_name = evaluation_result.get('batch_name', 'Unknown Experiment')
        agg_scores = evaluation_result.get('aggregated_scores', {})
        
        if not agg_scores:
            # Create empty plot with message
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
            # Create empty plot with message
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
    
    # =============================================================================
    # COMPREHENSIVE REPORT GENERATION
    # =============================================================================
    
    def create_experiment_report(self, evaluations: List[Dict], report_name: str = "experiment_report"):
        """
        Generate comprehensive visualization report with multiple plot types.
        
        Creates an organized collection of visualizations with an HTML index
        for easy navigation and analysis.
        
        Args:
            evaluations: List of evaluation results
            report_name: Name for the report directory
            
        Returns:
            tuple: (plots_dict, report_directory_path)
        """
        logger.info(f"Creating comprehensive experiment report: {report_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.plots_base_dir, "comprehensive_reports", f"{report_name}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        plots = {}
        
        try:
            # Generate metric comparison plots
            plot_configs = [
                ('f1_score', 'F1 Score'),
                ('semantic_similarity', 'Semantic Similarity'),
                ('precision', 'Precision'),
                ('recall', 'Recall'),
                ('exact_match', 'Exact Match')
            ]
            
            for metric, title in plot_configs:
                try:
                    fig = self.plot_metric_comparison(
                        evaluations, metric, 
                        save_path=os.path.join(report_dir, f"{metric}_comparison.html")
                    )
                    plots[f'{metric}_comparison'] = fig
                except Exception as e:
                    logger.error(f"Error creating {metric} comparison plot: {e}")
            
            # Create radar chart comparison
            try:
                radar_fig = self.plot_model_comparison_radar(
                    evaluations,
                    save_path=os.path.join(report_dir, "model_comparison_radar.html")
                )
                plots['radar_comparison'] = radar_fig
            except Exception as e:
                logger.error(f"Error creating radar chart: {e}")
            
            # Create prompt comparison plots
            try:
                prompt_fig1 = self.plot_prompt_comparison(
                    evaluations, group_by_model=True,
                    save_path=os.path.join(report_dir, "prompt_effects_by_model.html")
                )
                plots['prompt_by_model'] = prompt_fig1
                
                prompt_fig2 = self.plot_prompt_comparison(
                    evaluations, group_by_model=False,
                    save_path=os.path.join(report_dir, "model_performance_by_prompt.html")
                )
                plots['model_by_prompt'] = prompt_fig2
            except Exception as e:
                logger.error(f"Error creating prompt comparison plots: {e}")
            
            # Create comprehensive index HTML file
            self.create_report_index(report_dir, plots, evaluations)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
        
        logger.info(f"Comprehensive report generated at: {report_dir}")
        return plots, report_dir
    
    def create_report_index(self, report_dir: str, plots: Dict, evaluations: List[Dict]):
        """
        Create an HTML index file for comprehensive navigation of the report.
        
        Args:
            report_dir: Directory containing the report
            plots: Dictionary of generated plots
            evaluations: Original evaluation data for statistics
        """
        logger.info("Creating report index HTML")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>XAI Explanation Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .plot-link {{ display: inline-block; margin: 10px; padding: 10px 20px; 
                             background: #007bff; color: white; text-decoration: none; 
                             border-radius: 5px; }}
                .plot-link:hover {{ background: #0056b3; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .stat-box {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>XAI Explanation Evaluation Report</h1>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Experiments:</strong> {len(evaluations)}</p>
                <p><strong>Plots Generated:</strong> {len(plots)}</p>
                <p><strong>Report Directory:</strong> {os.path.basename(report_dir)}</p>
            </div>
            
            <h2>Generated Plots</h2>
            <div>
        """
        
        # Add links to plots with readable names
        plot_names = {
            'f1_score_comparison': 'F1 Score Comparison',
            'semantic_similarity_comparison': 'Semantic Similarity Comparison',
            'precision_comparison': 'Precision Comparison',
            'recall_comparison': 'Recall Comparison',
            'exact_match_comparison': 'Exact Match Comparison',
            'radar_comparison': 'Model Radar Comparison',
            'prompt_by_model': 'Prompt Effects by Model',
            'model_by_prompt': 'Model Performance by Prompt'
        }
        
        for plot_key in plots.keys():
            plot_name = plot_names.get(plot_key, plot_key.replace('_', ' ').title())
            html_content += f'<a href="{plot_key}.html" class="plot-link">{plot_name}</a>'
        
        html_content += """
            </div>
            
            <h2>Experiment Statistics</h2>
            <div class="stats">
        """
        
        # Add experiment statistics
        if evaluations:
            total_samples = sum(eval_result.get('num_samples', 0) for eval_result in evaluations)
            total_valid = sum(eval_result.get('num_valid_evaluations', 0) for eval_result in evaluations)
            
            # Find best performing experiment
            best_f1 = max(evaluations, 
                         key=lambda x: x.get('aggregated_scores', {}).get('f1_score', {}).get('mean', 0))
            
            html_content += f"""
                <div class="stat-box">
                    <h3>Overall Statistics</h3>
                    <p>Total Samples: {total_samples}</p>
                    <p>Valid Evaluations: {total_valid}</p>
                    <p>Success Rate: {(total_valid/total_samples*100):.1f}%</p>
                </div>
                
                <div class="stat-box">
                    <h3>Best Performance</h3>
                    <p>Experiment: {best_f1.get('batch_name', 'Unknown')}</p>
                    <p>F1 Score: {best_f1.get('aggregated_scores', {}).get('f1_score', {}).get('mean', 0):.4f}</p>
                    <p>Model: {best_f1.get('model_name', 'Unknown')}</p>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        index_path = os.path.join(report_dir, 'index.html')
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report index created: {index_path}")
    
    # =============================================================================
    # DATA EXPORT AND SUMMARY
    # =============================================================================
    
    def create_summary_csv(self, evaluations: List[Dict], save_path) -> str:
        """
        Create a CSV summary of all evaluation results for further analysis.
        
        Args:
            evaluations: List of evaluation results
            save_path: Optional custom save path
            
        Returns:
            str: Path to saved CSV file
        """
        logger.info(f"Creating summary CSV for {len(evaluations)} evaluations")
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(Config.OUTPUTS_DIR, f"experiment_summary_{timestamp}.csv")
        
        summary_data = []
        
        # Extract key information from each evaluation
        for result in evaluations:
            agg_scores = result.get('aggregated_scores', {})
            
            summary_row = {
                'experiment_name': result.get('batch_name', 'Unknown'),
                'model_name': result.get('model_name', 'Unknown'),
                'model_type': result.get('model_type', 'Unknown'),
                'prompt_key': result.get('prompt_key', 'Unknown'),
                'sample_size': result.get('sample_size', 0),
                'valid_evaluations': result.get('num_valid_evaluations', 0),
                'skipped_na': result.get('num_skipped_na', 0),
                'timestamp': result.get('timestamp', '')
            }
            
            # Add key metrics with mean and standard deviation
            for metric in ['f1_score', 'semantic_similarity', 'precision', 'recall', 'exact_match']:
                if metric in agg_scores:
                    summary_row[f'{metric}_mean'] = round(agg_scores[metric]['mean'], 4)
                    summary_row[f'{metric}_std'] = round(agg_scores[metric]['std'], 4)
                else:
                    summary_row[f'{metric}_mean'] = None
                    summary_row[f'{metric}_std'] = None
            
            summary_data.append(summary_row)
        
        # Create and save DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        summary_df.to_csv(save_path, index=False)
        
        logger.info(f"Summary CSV saved to: {save_path}")
        return save_path