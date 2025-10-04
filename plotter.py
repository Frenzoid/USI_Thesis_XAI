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
    Modern plotting system for evaluation results with aggregated summary support.
    
    Features:
    - Visualizes aggregated summary comparisons
    - Dynamic custom metrics plotting
    - Prompt vs model performance analysis
    - Pruning statistics visualization
    - Interactive dashboards
    """
    
    def __init__(self):
        """Initialize plotting runner with visualization framework"""
        self.visualization_framework = VisualizationFramework()
        logger.info("PlottingRunner initialized")
    
    def find_evaluation_files(self) -> List[str]:
        """Find all evaluation result files for plotting"""
        pattern = os.path.join(Config.EVALUATIONS_DIR, "evaluation_*.json")
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} evaluation files")
        return files
    
    def load_evaluation_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load evaluation results from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded evaluation results from: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading evaluation results from {file_path}: {e}")
            return None
    
    def load_aggregated_summary(self) -> Optional[Dict[str, Any]]:
        """Load the aggregated summary file"""
        summary_path = os.path.join(Config.EVALUATIONS_DIR, "aggregated_summary.json")
        
        if not os.path.exists(summary_path):
            logger.warning("Aggregated summary not found. Run evaluation first.")
            return None
        
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
            logger.info("Loaded aggregated summary")
            return data
        except Exception as e:
            logger.error(f"Error loading aggregated summary: {e}")
            return None
    
    def create_aggregated_summary_plots(self) -> Optional[str]:
        """
        Create comprehensive plots from aggregated summary.
        
        Returns:
            str: Path to summary dashboard HTML file
        """
        logger.info("Creating plots from aggregated summary")
        
        summary = self.load_aggregated_summary()
        if not summary:
            logger.error("Cannot create plots without aggregated summary")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(Config.PLOTS_DIR, f"aggregated_dashboard_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        plots = {}
        
        try:
            # Plot 1: Prompt comparison heatmap
            prompt_heatmap = self.visualization_framework.create_prompt_comparison_heatmap(
                summary.get('by_prompt', {}),
                save_path=os.path.join(output_dir, "prompt_comparison_heatmap.html")
            )
            plots['prompt_heatmap'] = prompt_heatmap
            
            # Plot 2: Model comparison heatmap
            model_heatmap = self.visualization_framework.create_model_comparison_heatmap(
                summary.get('by_model', {}),
                save_path=os.path.join(output_dir, "model_comparison_heatmap.html")
            )
            plots['model_heatmap'] = model_heatmap
            
            # Plot 3: Prompt+Model combination performance
            combo_plot = self.visualization_framework.create_combination_performance_plot(
                summary.get('by_prompt_model', {}),
                save_path=os.path.join(output_dir, "prompt_model_combinations.html")
            )
            plots['combo_plot'] = combo_plot
            
            # Plot 4: Best performers summary
            best_performers_plot = self.visualization_framework.create_best_performers_plot(
                summary.get('overall_best_performers', {}),
                save_path=os.path.join(output_dir, "best_performers.html")
            )
            plots['best_performers'] = best_performers_plot
            
            # Plot 5: Metric distribution across all experiments
            metric_dist_plot = self.visualization_framework.create_metric_distribution_plot(
                summary.get('by_prompt', {}),
                summary.get('by_model', {}),
                save_path=os.path.join(output_dir, "metric_distributions.html")
            )
            plots['metric_dist'] = metric_dist_plot
            
            # Create dashboard index
            dashboard_path = self._create_dashboard_index(output_dir, plots, summary)
            
            logger.info(f"Created {len(plots)} plots in dashboard: {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            logger.error(f"Error creating aggregated summary plots: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_dashboard_index(self, output_dir: str, plots: Dict, summary: Dict) -> str:
        """Create interactive dashboard index HTML"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Dashboard</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                       background: #f5f7fa; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; text-align: center; }}
                .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                              gap: 20px; margin: 30px 0; }}
                .stat-card {{ background: white; padding: 25px; border-radius: 10px; 
                             box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .stat-value {{ font-size: 2.5em; font-weight: bold; color: #667eea; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
                .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                             gap: 20px; margin: 30px 0; }}
                .plot-card {{ background: white; padding: 20px; border-radius: 10px; 
                             box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: transform 0.2s; }}
                .plot-card:hover {{ transform: translateY(-5px); box-shadow: 0 4px 20px rgba(0,0,0,0.15); }}
                .plot-link {{ display: block; padding: 15px; background: #667eea; color: white; 
                             text-decoration: none; border-radius: 5px; text-align: center; 
                             margin-top: 15px; }}
                .plot-link:hover {{ background: #764ba2; }}
                h2 {{ color: #333; margin: 30px 0 20px 0; padding-bottom: 10px; 
                     border-bottom: 2px solid #667eea; }}
                .best-performer {{ background: #f0f9ff; border-left: 4px solid #667eea; 
                                  padding: 15px; margin: 10px 0; }}
                .timestamp {{ color: #888; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Evaluation Dashboard</h1>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="container">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{summary.get('total_experiments', 0)}</div>
                        <div class="stat-label">Total Experiments</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{summary.get('unique_prompts', 0)}</div>
                        <div class="stat-label">Unique Prompts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{summary.get('unique_models', 0)}</div>
                        <div class="stat-label">Unique Models</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(plots)}</div>
                        <div class="stat-label">Visualizations</div>
                    </div>
                </div>
                
                <h2>🏆 Best Performers</h2>
        """
        
        best = summary.get('overall_best_performers', {})
        for metric_key, performer in best.items():
            if performer:
                metric_name = metric_key.replace('by_', '').replace('_', ' ').title()
                html_content += f"""
                <div class="best-performer">
                    <strong>{metric_name}:</strong> {performer.get('score', 0):.3f}<br>
                    <small>Prompt: {performer.get('prompt', 'unknown')} | 
                    Model: {performer.get('model', 'unknown')}</small>
                </div>
                """
        
        html_content += """
                <h2>📈 Visualizations</h2>
                <div class="plot-grid">
        """
        
        plot_info = {
            'prompt_heatmap': ('Prompt Comparison Heatmap', 'Compare all prompts across metrics'),
            'model_heatmap': ('Model Comparison Heatmap', 'Compare all models across metrics'),
            'combo_plot': ('Prompt+Model Combinations', 'Performance of each combination'),
            'best_performers': ('Best Performers Breakdown', 'Top performers per metric'),
            'metric_dist': ('Metric Distributions', 'Statistical distribution of metrics')
        }
        
        for plot_key, (title, description) in plot_info.items():
            if plot_key in plots:
                filename = f"{plot_key}.html" if not plot_key.endswith('.html') else plot_key
                html_content += f"""
                <div class="plot-card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                    <a href="{filename}" class="plot-link">View Plot →</a>
                </div>
                """
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        dashboard_path = os.path.join(output_dir, "index.html")
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard index created: {dashboard_path}")
        return dashboard_path
    
    def create_all_plots(self) -> Optional[Dict[str, Any]]:
        """
        Create all available plots including aggregated summary dashboard.
        
        Returns:
            dict: Paths to generated plots and dashboards
        """
        logger.info("Creating all available plots")
        
        result = {
            'aggregated_dashboard': None,
            'individual_plots': []
        }
        
        # Create aggregated summary dashboard
        dashboard_path = self.create_aggregated_summary_plots()
        if dashboard_path:
            result['aggregated_dashboard'] = dashboard_path
            logger.info(f"Aggregated dashboard: {dashboard_path}")
        
        logger.info(f"Plot generation complete")
        return result