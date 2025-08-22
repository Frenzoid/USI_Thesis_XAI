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
    """Comprehensive visualization system for experiment results with auto-generation"""
    
    def __init__(self):
        # Create plots directory structure
        self.plots_base_dir = Config.PLOTS_DIR
        self.ensure_directory_structure()
        
        logger.info("VisualizationFramework initialized")
    
    def ensure_directory_structure(self):
        """Ensure proper directory structure for plots"""
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
    
    def plot_metric_comparison(self, evaluations: List[Dict], metric: str = 'f1_score',
                              save_path: str = None, show_error_bars: bool = True) -> go.Figure:
        """Plot comparison of a specific metric across evaluations with enhanced styling"""
        logger.info(f"Creating metric comparison plot for '{metric}' across {len(evaluations)} evaluations")
        
        names = []
        means = []
        stds = []
        colors = []
        
        # Color scheme based on model type
        color_map = {
            'open_source': '#1f77b4',  # Blue
            'api': '#ff7f0e',          # Orange
            'gpt': '#2ca02c',          # Green
            'gemini': '#d62728',       # Red
            'claude': '#9467bd'        # Purple
        }
        
        for eval_result in evaluations:
            name = eval_result.get('batch_name', 'Unknown')
            model_type = eval_result.get('model_type', 'unknown')
            model_name = eval_result.get('model_name', '')
            
            names.append(name)
            agg_scores = eval_result['aggregated_scores']
            
            if metric in agg_scores:
                means.append(agg_scores[metric]['mean'])
                stds.append(agg_scores[metric]['std'] if show_error_bars else 0)
            else:
                means.append(0)
                stds.append(0)
            
            # Determine color
            if 'gpt' in model_name.lower():
                colors.append(color_map.get('gpt', color_map['api']))
            elif 'gemini' in model_name.lower():
                colors.append(color_map.get('gemini', color_map['api']))
            elif 'claude' in model_name.lower():
                colors.append(color_map.get('claude', color_map['api']))
            else:
                colors.append(color_map.get(model_type, '#666666'))
        
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
        
        fig.update_layout(
            title=f'{metric.replace("_", " ").title()} Comparison Across Models',
            xaxis_title='Experiment',
            yaxis_title=metric.replace('_', ' ').title(),
            showlegend=False,
            xaxis_tickangle=-45,
            height=600,
            margin=dict(b=150),  # More space for rotated labels
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black")
        )
        
        # Add horizontal line for average
        if means:
            avg_score = np.mean(means)
            fig.add_hline(y=avg_score, line_dash="dash", line_color="gray",
                         annotation_text=f"Average: {avg_score:.3f}")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved {metric} comparison plot to: {save_path}")
        
        return fig
    
    def plot_model_comparison_radar(self, evaluations: List[Dict], 
                                   metrics: List[str] = None, save_path: str = None,
                                   max_models: int = 8) -> go.Figure:
        """Create radar chart comparing multiple models across metrics with improved styling"""
        logger.info(f"Creating radar chart for {len(evaluations)} evaluations")
        
        if metrics is None:
            metrics = ['f1_score', 'semantic_similarity', 'precision', 'recall', 'exact_match']
        
        fig = go.Figure()
        
        # Limit number of models for readability
        evaluations_to_plot = evaluations[:max_models] if len(evaluations) > max_models else evaluations
        
        if len(evaluations) > max_models:
            logger.warning(f"Limiting radar chart to {max_models} models for readability")
        
        colors = px.colors.qualitative.Set3[:len(evaluations_to_plot)]
        
        for i, eval_result in enumerate(evaluations_to_plot):
            name = eval_result.get('batch_name', 'Unknown')
            model_name = eval_result.get('model_name', '')
            prompt_key = eval_result.get('prompt_key', '')
            
            # Create shorter label
            short_name = f"{model_name}-{prompt_key}" if len(name) > 20 else name
            
            agg_scores = eval_result['aggregated_scores']
            
            values = []
            labels = []
            
            for metric in metrics:
                if metric in agg_scores:
                    values.append(agg_scores[metric]['mean'])
                    labels.append(metric.replace('_', ' ').title())
            
            # Close the radar chart
            if values:
                values.append(values[0])
                labels.append(labels[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=short_name,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.6
            ))
        
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
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved radar chart to: {save_path}")
        
        return fig
    
    def plot_prompt_comparison(self, evaluations: List[Dict], group_by_model: bool = True,
                              save_path: str = None) -> go.Figure:
        """Create comparison plot focused on prompt engineering effects"""
        logger.info(f"Creating prompt comparison plot, group_by_model={group_by_model}")
        
        if group_by_model:
            # Group by model, compare prompts
            model_groups = {}
            for eval_result in evaluations:
                model_name = eval_result.get('model_name', 'Unknown')
                if model_name not in model_groups:
                    model_groups[model_name] = []
                model_groups[model_name].append(eval_result)
            
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
            # Group by prompt, compare models
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
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved prompt comparison plot to: {save_path}")
        
        return fig
    
    def create_experiment_report(self, evaluations: List[Dict], report_name: str = "experiment_report"):
        """Generate comprehensive visualization report with organized structure"""
        logger.info(f"Creating comprehensive experiment report: {report_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.plots_base_dir, "comprehensive_reports", f"{report_name}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        plots = {}
        
        try:
            # Generate all plot types
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
            
            # Radar chart
            try:
                radar_fig = self.plot_model_comparison_radar(
                    evaluations,
                    save_path=os.path.join(report_dir, "model_comparison_radar.html")
                )
                plots['radar_comparison'] = radar_fig
            except Exception as e:
                logger.error(f"Error creating radar chart: {e}")
            
            # Prompt comparison plots
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
            
            # Create index HTML file
            self.create_report_index(report_dir, plots, evaluations)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
        
        logger.info(f"Comprehensive report generated at: {report_dir}")
        return plots, report_dir
    
    def create_report_index(self, report_dir: str, plots: Dict, evaluations: List[Dict]):
        """Create an index.html file for the report"""
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
        
        # Add links to plots
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
            
            # Get best performing experiment
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
    
    def create_summary_csv(self, evaluations: List[Dict], save_path: str = None) -> str:
        """Create a CSV summary of all evaluation results"""
        logger.info(f"Creating summary CSV for {len(evaluations)} evaluations")
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(Config.RESULTS_DIR, f"experiment_summary_{timestamp}.csv")
        
        summary_data = []
        
        for result in evaluations:
            agg_scores = result.get('aggregated_scores', {})
            proc_stats = result.get('processing_stats', {})
            
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
            
            # Add key metrics
            for metric in ['f1_score', 'semantic_similarity', 'precision', 'recall', 'exact_match']:
                if metric in agg_scores:
                    summary_row[f'{metric}_mean'] = round(agg_scores[metric]['mean'], 4)
                    summary_row[f'{metric}_std'] = round(agg_scores[metric]['std'], 4)
                else:
                    summary_row[f'{metric}_mean'] = None
                    summary_row[f'{metric}_std'] = None
            
            # Add processing stats
            if proc_stats:
                summary_row['avg_processing_time'] = round(proc_stats.get('avg_processing_time', 0), 2)
                summary_row['total_processing_time'] = round(proc_stats.get('total_processing_time', 0), 2)
            
            summary_data.append(summary_row)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_path, index=False)
        
        logger.info(f"Summary CSV saved to: {save_path}")
        return save_path
