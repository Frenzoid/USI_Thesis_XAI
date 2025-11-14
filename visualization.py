#!/usr/bin/env python3
"""
Visualization System for Aggregated Evaluation Results

Creates comprehensive visualizations from aggregated_summary.json:
- Radar charts comparing models across metrics
- Bar charts for metric comparisons
- Heatmaps showing performance patterns
- HTML dashboard for interactive exploration

Works with simplified aggregated summary format.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import Config

# Setup logging
logger = logging.getLogger(__name__)

# Standard metrics that appear across most evaluations
STANDARD_METRICS = {
    'exact_match', 'precision', 'recall', 'f1_score', 'jaccard',
    'bleu', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'semantic_similarity'
}

# =============================================================================
# DATA NAVIGATION
# =============================================================================

class AggregatedDataNavigator:
    """
    Navigator for simplified aggregated summary data.
    
    Provides efficient access to the nested structure:
    results -> setup -> prompt -> model -> metrics
    """
    
    def __init__(self, json_path: str):
        """
        Initialize navigator with aggregated summary data.
        
        Args:
            json_path: Path to aggregated_summary.json file
        """
        self.json_path = json_path
        self.data = self._load_data(json_path)
        self.setups = list(self.data.get('results', {}).keys())
        
        # Cache structure for quick access
        self.setup_prompts = {}
        self.setup_models = {}
        self.setup_custom_metrics = {}
        
        self._cache_structure()
        
        logger.info(f"Loaded aggregated data with {len(self.setups)} setups")
    
    def _load_data(self, json_path: str) -> Dict:
        """Load JSON data from file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded aggregated data from: {json_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Aggregated summary not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in aggregated summary: {e}")
            raise
    
    def _cache_structure(self):
        """Build quick lookup caches for setups, prompts, and models."""
        data_source = self.data.get('results', {})
        
        for setup in self.setups:
            # Cache prompts for this setup
            self.setup_prompts[setup] = list(data_source.get(setup, {}).keys())
            
            # Cache all unique models across prompts in this setup
            models = set()
            for prompt_data in data_source.get(setup, {}).values():
                models.update(prompt_data.keys())
            self.setup_models[setup] = sorted(list(models))
            
            # Cache custom metrics for this setup
            self.setup_custom_metrics[setup] = self._identify_custom_metrics(setup)
        
        logger.debug(f"Cached structure: {len(self.setup_prompts)} setups indexed")
    
    def _identify_custom_metrics(self, setup: str) -> Set[str]:
        """
        Identify custom metrics for a setup (non-standard metrics).
        
        Args:
            setup: Setup name
            
        Returns:
            set: Custom metric names
        """
        all_metrics = set()
        
        setup_data = self.data['results'].get(setup, {})
        for prompt_data in setup_data.values():
            for model_data in prompt_data.values():
                metrics = model_data.get('metrics', {})
                all_metrics.update(metrics.keys())
        
        # Custom metrics = all metrics - standard metrics
        custom_metrics = all_metrics - STANDARD_METRICS
        
        return custom_metrics
    
    def get_all_unique_models(self) -> List[str]:
        """Get all unique models across all setups."""
        all_models = set()
        for models in self.setup_models.values():
            all_models.update(models)
        return sorted(list(all_models))
    
    def get_prompts_for_setup(self, setup: str) -> List[str]:
        """Get all prompts for a setup."""
        return self.setup_prompts.get(setup, [])
    
    def get_models_for_setup(self, setup: str) -> List[str]:
        """Get all unique models that ran on a setup."""
        return self.setup_models.get(setup, [])
    
    def get_custom_metrics_for_setup(self, setup: str) -> Set[str]:
        """Get custom (non-standard) metrics for a setup."""
        return self.setup_custom_metrics.get(setup, set())
    
    def get_metric_value(self, setup: str, prompt: str, model: str, 
                        metric: str) -> Optional[float]:
        """
        Get a specific metric value for a setup-prompt-model combination.
        
        Args:
            setup: Setup name
            prompt: Prompt name
            model: Model name
            metric: Metric name
            
        Returns:
            float: Metric value, or None if not found
        """
        try:
            value = self.data['results'][setup][prompt][model]['metrics'].get(metric)
            return value
        except (KeyError, TypeError):
            return None
    
    def get_all_metrics_for_model(self, setup: str, prompt: str, 
                                  model: str) -> Dict[str, float]:
        """
        Get all metrics for a specific setup-prompt-model combination.
        
        Args:
            setup: Setup name
            prompt: Prompt name
            model: Model name
            
        Returns:
            dict: Metric name -> value mapping
        """
        try:
            return self.data['results'][setup][prompt][model]['metrics']
        except (KeyError, TypeError):
            return {}

# =============================================================================
# SETUP-LEVEL PLOTTING
# =============================================================================

class SetupPlotter:
    """
    Creates visualizations for a single setup.
    
    Generates:
    - Radar charts comparing models across metrics
    - Bar charts for individual metrics
    - Heatmaps showing performance patterns
    """
    
    def __init__(self, setup_name: str, navigator: AggregatedDataNavigator, 
                 output_dir: str):
        """
        Initialize plotter for a specific setup.
        
        Args:
            setup_name: Name of the setup to visualize
            navigator: Data navigator instance
            output_dir: Directory to save plots
        """
        self.setup_name = setup_name
        self.navigator = navigator
        self.output_dir = output_dir
        
        # Setup-specific data
        self.prompts = navigator.get_prompts_for_setup(setup_name)
        self.models = navigator.get_models_for_setup(setup_name)
        self.custom_metrics = navigator.get_custom_metrics_for_setup(setup_name)
        
        # Create output subdirectory
        self.setup_dir = os.path.join(output_dir, setup_name)
        os.makedirs(self.setup_dir, exist_ok=True)
        
        logger.info(f"SetupPlotter initialized for '{setup_name}': "
                   f"{len(self.prompts)} prompts, {len(self.models)} models")
    
    def plot_all(self) -> List[str]:
        """
        Generate all visualizations for this setup.
        
        Returns:
            list: Paths to generated plot files
        """
        plot_files = []
        
        # For each prompt, create visualizations
        for prompt in self.prompts:
            logger.info(f"Generating plots for {self.setup_name}/{prompt}")
            
            # Radar chart for standard metrics
            radar_file = self._plot_radar_chart(prompt, STANDARD_METRICS)
            if radar_file:
                plot_files.append(radar_file)
            
            # Bar charts for top metrics
            bar_files = self._plot_metric_bars(prompt)
            plot_files.extend(bar_files)
            
            # Heatmap for all metrics
            heatmap_file = self._plot_heatmap(prompt)
            if heatmap_file:
                plot_files.append(heatmap_file)
            
            # Custom metrics if available
            if self.custom_metrics:
                custom_radar_file = self._plot_radar_chart(prompt, self.custom_metrics, 
                                                          suffix='_custom')
                if custom_radar_file:
                    plot_files.append(custom_radar_file)
        
        logger.info(f"Generated {len(plot_files)} plots for setup '{self.setup_name}'")
        return plot_files
    
    def _plot_radar_chart(self, prompt: str, metrics: Set[str], 
                         suffix: str = '') -> Optional[str]:
        """
        Create radar chart comparing models across metrics.
        
        Args:
            prompt: Prompt name
            metrics: Set of metrics to include
            suffix: Optional suffix for filename
            
        Returns:
            str: Path to saved plot, or None if not enough data
        """
        # Collect data for each model
        model_data = {}
        available_metrics = []
        
        for model in self.models:
            all_metrics = self.navigator.get_all_metrics_for_model(
                self.setup_name, prompt, model
            )
            
            # Filter to requested metrics that have values
            values = {m: all_metrics.get(m) for m in metrics if m in all_metrics}
            
            if values:
                model_data[model] = values
                # Track which metrics have data
                if not available_metrics:
                    available_metrics = list(values.keys())
        
        if not model_data or not available_metrics:
            logger.debug(f"No data for radar chart: {self.setup_name}/{prompt}")
            return None
        
        # Ensure all models have same metrics (fill missing with 0)
        for model in model_data:
            for metric in available_metrics:
                if metric not in model_data[model]:
                    model_data[model][metric] = 0.0
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model, metrics_dict in model_data.items():
            values = [metrics_dict[m] for m in available_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(f"{self.setup_name} - {prompt}\nModel Comparison", 
                    size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        # Save
        filename = f"{prompt}_radar{suffix}.png"
        filepath = os.path.join(self.setup_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved radar chart: {filepath}")
        return filepath
    
    def _plot_metric_bars(self, prompt: str, top_n: int = 5) -> List[str]:
        """
        Create bar charts for top N most important metrics.
        
        Args:
            prompt: Prompt name
            top_n: Number of metrics to visualize
            
        Returns:
            list: Paths to saved plots
        """
        # Priority metrics to visualize
        priority_metrics = ['f1_score', 'semantic_similarity', 'bleu', 
                           'rouge1_f', 'exact_match', 'precision', 'recall']
        
        # Get available metrics for this prompt
        available_metrics = set()
        for model in self.models:
            metrics = self.navigator.get_all_metrics_for_model(
                self.setup_name, prompt, model
            )
            available_metrics.update(metrics.keys())
        
        # Select metrics to plot (prioritize important ones)
        metrics_to_plot = []
        for metric in priority_metrics:
            if metric in available_metrics:
                metrics_to_plot.append(metric)
                if len(metrics_to_plot) >= top_n:
                    break
        
        # Add custom metrics if we need more
        if len(metrics_to_plot) < top_n:
            remaining = self.custom_metrics & available_metrics
            metrics_to_plot.extend(list(remaining)[:top_n - len(metrics_to_plot)])
        
        if not metrics_to_plot:
            return []
        
        # Create bar chart
        fig, axes = plt.subplots(1, len(metrics_to_plot), 
                                figsize=(5 * len(metrics_to_plot), 6))
        
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        plot_files = []
        
        for ax, metric in zip(axes, metrics_to_plot):
            # Collect values for this metric
            model_names = []
            values = []
            
            for model in self.models:
                value = self.navigator.get_metric_value(
                    self.setup_name, prompt, model, metric
                )
                if value is not None:
                    model_names.append(model)
                    values.append(value)
            
            if not values:
                continue
            
            # Create bar chart
            bars = ax.bar(range(len(model_names)), values, alpha=0.8)
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f"{self.setup_name} - {prompt}\nMetric Comparison", 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save
        filename = f"{prompt}_bars.png"
        filepath = os.path.join(self.setup_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_files.append(filepath)
        logger.debug(f"Saved bar chart: {filepath}")
        
        return plot_files
    
    def _plot_heatmap(self, prompt: str) -> Optional[str]:
        """
        Create heatmap showing all metrics for all models.
        
        Args:
            prompt: Prompt name
            
        Returns:
            str: Path to saved plot, or None if not enough data
        """
        # Collect all metrics for all models
        data_matrix = []
        metric_names = []
        model_names = []
        
        # First pass: collect all unique metrics
        all_metrics = set()
        for model in self.models:
            metrics = self.navigator.get_all_metrics_for_model(
                self.setup_name, prompt, model
            )
            all_metrics.update(metrics.keys())
        
        if not all_metrics:
            return None
        
        metric_names = sorted(list(all_metrics))
        
        # Second pass: build data matrix
        for model in self.models:
            metrics = self.navigator.get_all_metrics_for_model(
                self.setup_name, prompt, model
            )
            
            row = [metrics.get(m, 0.0) for m in metric_names]
            
            # Only include if model has some data
            if any(v > 0 for v in row):
                data_matrix.append(row)
                model_names.append(model)
        
        if not data_matrix:
            return None
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(metric_names) * 0.8), 
                                       max(6, len(model_names) * 0.6)))
        
        im = ax.imshow(data_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        # Add value annotations
        for i in range(len(model_names)):
            for j in range(len(metric_names)):
                value = data_matrix[i][j]
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", 
                             color="black" if value > 0.5 else "white",
                             fontsize=8)
        
        ax.set_title(f"{self.setup_name} - {prompt}\nPerformance Heatmap", 
                    fontsize=14, pad=15)
        plt.tight_layout()
        
        # Save
        filename = f"{prompt}_heatmap.png"
        filepath = os.path.join(self.setup_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved heatmap: {filepath}")
        return filepath

# =============================================================================
# VISUALIZATION FRAMEWORK
# =============================================================================

class VisualizationFramework:
    """
    Main framework for generating all visualizations from aggregated data.
    
    Orchestrates:
    - Setup-level plotting
    - HTML dashboard generation
    """
    
    def __init__(self, aggregated_json_path: str, output_dir: str = None):
        """
        Initialize visualization framework.
        
        Args:
            aggregated_json_path: Path to aggregated_summary.json
            output_dir: Output directory for plots (defaults to Config.PLOTS_DIR)
        """
        self.aggregated_json_path = aggregated_json_path
        self.output_dir = output_dir or Config.PLOTS_DIR
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize navigator
        self.navigator = AggregatedDataNavigator(aggregated_json_path)
        
        logger.info(f"VisualizationFramework initialized")
        logger.info(f"  Input: {aggregated_json_path}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Setups: {len(self.navigator.setups)}")
    
    def generate_all_visualizations(self) -> Dict[str, List[str]]:
        """
        Generate all visualizations for all setups.
        
        Returns:
            dict: Setup name -> list of plot file paths
        """
        logger.info("Generating all visualizations...")
        
        all_plots = {}
        
        # Generate plots for each setup
        for setup_name in self.navigator.setups:
            logger.info(f"Processing setup: {setup_name}")
            
            plotter = SetupPlotter(setup_name, self.navigator, self.output_dir)
            plot_files = plotter.plot_all()
            
            all_plots[setup_name] = plot_files
        
        # Generate HTML dashboard
        dashboard_path = self._generate_dashboard(all_plots)
        
        logger.info(f"Generated {sum(len(plots) for plots in all_plots.values())} total plots")
        logger.info(f"HTML Dashboard: {dashboard_path}")
        
        return all_plots
    
    def _generate_dashboard(self, all_plots: Dict[str, List[str]]) -> str:
        """
        Generate HTML dashboard with all visualizations.
        
        Args:
            all_plots: Setup -> plot files mapping
            
        Returns:
            str: Path to generated HTML file
        """
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Visualization Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .header p {
            margin: 5px 0;
            opacity: 0.9;
        }
        .setup-section {
            background: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .setup-section h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .plot-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .plot-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .plot-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .plot-title {
            padding: 12px;
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
            text-align: center;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            text-align: center;
            min-width: 150px;
            margin: 10px;
        }
        .stat-box .number {
            font-size: 36px;
            font-weight: bold;
            margin: 5px 0;
        }
        .stat-box .label {
            font-size: 14px;
            opacity: 0.9;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
"""
        
        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_plots = sum(len(plots) for plots in all_plots.values())
        all_models = self.navigator.get_all_unique_models()
        
        html_content += f"""
    <div class="header">
        <h1>🔬 Evaluation Visualization Dashboard</h1>
        <p>Generated: {timestamp}</p>
        <p>Setups: {len(all_plots)} | Total Visualizations: {total_plots}</p>
    </div>
"""
        
        # Statistics
        html_content += """
    <div class="stats">
"""
        html_content += f"""
        <div class="stat-box">
            <div class="number">{len(all_plots)}</div>
            <div class="label">Setups</div>
        </div>
        <div class="stat-box">
            <div class="number">{len(all_models)}</div>
            <div class="label">Models</div>
        </div>
        <div class="stat-box">
            <div class="number">{total_plots}</div>
            <div class="label">Visualizations</div>
        </div>
"""
        html_content += """
    </div>
"""
        
        # Setup sections
        for setup_name, plot_files in all_plots.items():
            if not plot_files:
                continue
            
            html_content += f"""
    <div class="setup-section">
        <h2>{setup_name}</h2>
        <div class="plot-grid">
"""
            
            for plot_file in plot_files:
                # Get relative path from output_dir
                rel_path = os.path.relpath(plot_file, self.output_dir)
                plot_name = os.path.splitext(os.path.basename(plot_file))[0]
                plot_name = plot_name.replace('_', ' ').title()
                
                html_content += f"""
            <div class="plot-card">
                <div class="plot-title">{plot_name}</div>
                <img src="{rel_path}" alt="{plot_name}">
            </div>
"""
            
            html_content += """
        </div>
    </div>
"""
        
        # Footer
        html_content += """
    <div class="footer">
        <p>Generated by XAI Evaluation System Visualization Framework</p>
    </div>
</body>
</html>
"""
        
        # Save HTML file
        dashboard_path = os.path.join(self.output_dir, "dashboard.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML dashboard saved: {dashboard_path}")
        return dashboard_path