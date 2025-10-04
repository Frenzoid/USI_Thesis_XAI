import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import Config
from utils import setup_logging

logger = setup_logging("visualization")

class VisualizationFramework:
    """
    Modern visualization system for evaluation results.
    
    Features:
    - Dynamic custom metrics visualization
    - Heatmap comparisons for prompts and models
    - Interactive combination analysis
    - Statistical distributions
    - Best performers breakdown
    """
    
    def __init__(self):
        """Initialize visualization framework"""
        self.plots_base_dir = Config.PLOTS_DIR
        os.makedirs(self.plots_base_dir, exist_ok=True)
        logger.info("VisualizationFramework initialized")
    
    def create_prompt_comparison_heatmap(self, by_prompt: Dict[str, Any], 
                                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create heatmap comparing all prompts across all metrics.
        
        Args:
            by_prompt: Aggregated data grouped by prompt
            save_path: Optional path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive heatmap
        """
        logger.info(f"Creating prompt comparison heatmap for {len(by_prompt)} prompts")
        
        if not by_prompt:
            fig = go.Figure()
            fig.add_annotation(text="No prompt data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Extract all metrics
        all_metrics = set()
        for prompt_data in by_prompt.values():
            metrics = prompt_data.get('metrics', {})
            all_metrics.update(metrics.keys())
        
        all_metrics = sorted(list(all_metrics))
        prompts = sorted(list(by_prompt.keys()))
        
        # Build heatmap matrix
        heatmap_data = []
        for prompt in prompts:
            row = []
            for metric in all_metrics:
                metric_data = by_prompt[prompt].get('metrics', {}).get(metric, {})
                mean_value = metric_data.get('mean', 0.0) if isinstance(metric_data, dict) else 0.0
                row.append(mean_value)
            heatmap_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[m.replace('_', ' ').title() for m in all_metrics],
            y=prompts,
            colorscale='RdYlGn',
            text=[[f'{val:.3f}' for val in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title='Prompt Performance Across All Metrics',
            xaxis_title='Metrics',
            yaxis_title='Prompts',
            height=max(400, len(prompts) * 50),
            width=max(800, len(all_metrics) * 80),
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved prompt heatmap to: {save_path}")
        
        return fig
    
    def create_model_comparison_heatmap(self, by_model: Dict[str, Any], 
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        Create heatmap comparing all models across all metrics.
        
        Args:
            by_model: Aggregated data grouped by model
            save_path: Optional path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive heatmap
        """
        logger.info(f"Creating model comparison heatmap for {len(by_model)} models")
        
        if not by_model:
            fig = go.Figure()
            fig.add_annotation(text="No model data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Extract all metrics
        all_metrics = set()
        for model_data in by_model.values():
            metrics = model_data.get('metrics', {})
            all_metrics.update(metrics.keys())
        
        all_metrics = sorted(list(all_metrics))
        models = sorted(list(by_model.keys()))
        
        # Build heatmap matrix
        heatmap_data = []
        for model in models:
            row = []
            for metric in all_metrics:
                metric_data = by_model[model].get('metrics', {}).get(metric, {})
                mean_value = metric_data.get('mean', 0.0) if isinstance(metric_data, dict) else 0.0
                row.append(mean_value)
            heatmap_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[m.replace('_', ' ').title() for m in all_metrics],
            y=models,
            colorscale='Viridis',
            text=[[f'{val:.3f}' for val in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title='Model Performance Across All Metrics',
            xaxis_title='Metrics',
            yaxis_title='Models',
            height=max(400, len(models) * 50),
            width=max(800, len(all_metrics) * 80),
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved model heatmap to: {save_path}")
        
        return fig
    
    def create_combination_performance_plot(self, by_prompt_model: Dict[str, Any],
                                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create bubble plot showing performance of prompt+model combinations.
        
        Args:
            by_prompt_model: Aggregated data grouped by prompt+model combination
            save_path: Optional path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive bubble plot
        """
        logger.info(f"Creating combination performance plot for {len(by_prompt_model)} combinations")
        
        if not by_prompt_model:
            fig = go.Figure()
            fig.add_annotation(text="No combination data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Extract data for bubble plot
        data_points = []
        for combo_key, combo_data in by_prompt_model.items():
            metrics = combo_data.get('metrics', {})
            
            # Get key metrics
            f1 = metrics.get('f1_score', {}).get('mean', 0) if isinstance(metrics.get('f1_score'), dict) else 0
            similarity = metrics.get('semantic_similarity', {}).get('mean', 0) if isinstance(metrics.get('semantic_similarity'), dict) else 0
            exact_match = metrics.get('exact_match', {}).get('mean', 0) if isinstance(metrics.get('exact_match'), dict) else 0
            
            # Parse prompt and model from experiments
            experiments = combo_data.get('experiments', [])
            if experiments:
                # Assume first experiment name format: setup_model_mode_prompt_...
                parts = experiments[0].split('_')
                model = parts[1] if len(parts) > 1 else 'unknown'
                prompt = parts[3] if len(parts) > 3 else 'unknown'
            else:
                # Fallback: parse from combo_key
                parts = combo_key.split('_')
                prompt = parts[0] if parts else 'unknown'
                model = parts[1] if len(parts) > 1 else 'unknown'
            
            data_points.append({
                'prompt': prompt,
                'model': model,
                'f1_score': f1,
                'semantic_similarity': similarity,
                'exact_match': exact_match,
                'num_experiments': combo_data.get('num_experiments', 1)
            })
        
        if not data_points:
            fig = go.Figure()
            fig.add_annotation(text="No valid data points", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(data_points)
        
        # Create bubble plot
        fig = px.scatter(df, 
                        x='f1_score', 
                        y='semantic_similarity',
                        size='exact_match',
                        color='model',
                        hover_data=['prompt', 'model', 'num_experiments'],
                        title='Prompt+Model Combination Performance',
                        labels={
                            'f1_score': 'F1 Score',
                            'semantic_similarity': 'Semantic Similarity',
                            'exact_match': 'Exact Match (size)'
                        },
                        size_max=30)
        
        fig.update_layout(
            height=600,
            width=1000,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved combination plot to: {save_path}")
        
        return fig
    
    def create_best_performers_plot(self, best_performers: Dict[str, Any],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create bar chart showing best performers across different metrics.
        
        Args:
            best_performers: Dictionary of best performers per metric
            save_path: Optional path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive bar chart
        """
        logger.info(f"Creating best performers plot")
        
        if not best_performers:
            fig = go.Figure()
            fig.add_annotation(text="No best performers data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        metrics = []
        scores = []
        prompts = []
        models = []
        
        for metric_key, performer_info in best_performers.items():
            if performer_info:
                metric_name = metric_key.replace('by_', '').replace('_', ' ').title()
                metrics.append(metric_name)
                scores.append(performer_info.get('score', 0))
                prompts.append(performer_info.get('prompt', 'unknown'))
                models.append(performer_info.get('model', 'unknown'))
        
        if not metrics:
            fig = go.Figure()
            fig.add_annotation(text="No valid performers data", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=scores,
            text=[f'{s:.3f}<br>{p}<br>{m}' for s, p, m in zip(scores, prompts, models)],
            textposition='auto',
            marker_color='lightcoral',
            hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<br>Prompt: %{customdata[0]}<br>Model: %{customdata[1]}<extra></extra>',
            customdata=list(zip(prompts, models))
        ))
        
        fig.update_layout(
            title='Best Performers by Metric',
            xaxis_title='Metric',
            yaxis_title='Score',
            height=500,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved best performers plot to: {save_path}")
        
        return fig
    
    def create_metric_distribution_plot(self, by_prompt: Dict[str, Any], 
                                       by_model: Dict[str, Any],
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        Create box plots showing distribution of key metrics across prompts and models.
        
        Args:
            by_prompt: Aggregated data grouped by prompt
            by_model: Aggregated data grouped by model
            save_path: Optional path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive box plot
        """
        logger.info("Creating metric distribution plot")
        
        # Key metrics to visualize
        key_metrics = ['f1_score', 'semantic_similarity', 'exact_match']
        
        # Create subplots: 2 rows (prompts, models), 3 columns (metrics)
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'{m.replace("_", " ").title()} by Prompt' for m in key_metrics] + 
                          [f'{m.replace("_", " ").title()} by Model' for m in key_metrics],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Plot distributions by prompt
        for col, metric in enumerate(key_metrics, 1):
            values_by_prompt = []
            prompt_names = []
            
            for prompt, prompt_data in by_prompt.items():
                metric_data = prompt_data.get('metrics', {}).get(metric, {})
                if isinstance(metric_data, dict) and 'values' in metric_data:
                    values_by_prompt.extend(metric_data['values'])
                    prompt_names.extend([prompt] * len(metric_data['values']))
            
            if values_by_prompt:
                fig.add_trace(
                    go.Box(x=prompt_names, y=values_by_prompt, name=f'{metric} (Prompt)', showlegend=False),
                    row=1, col=col
                )
        
        # Plot distributions by model
        for col, metric in enumerate(key_metrics, 1):
            values_by_model = []
            model_names = []
            
            for model, model_data in by_model.items():
                metric_data = model_data.get('metrics', {}).get(metric, {})
                if isinstance(metric_data, dict) and 'values' in metric_data:
                    values_by_model.extend(metric_data['values'])
                    model_names.extend([model] * len(metric_data['values']))
            
            if values_by_model:
                fig.add_trace(
                    go.Box(x=model_names, y=values_by_model, name=f'{metric} (Model)', showlegend=False),
                    row=2, col=col
                )
        
        fig.update_layout(
            title_text='Metric Distributions Across Prompts and Models',
            height=800,
            showlegend=False
        )
        
        # Update y-axes to have consistent range
        for i in range(1, 7):
            fig.update_yaxes(range=[0, 1], row=(i-1)//3+1, col=(i-1)%3+1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved metric distribution plot to: {save_path}")
        
        return fig