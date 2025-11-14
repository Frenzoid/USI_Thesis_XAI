#!/usr/bin/env python3
"""
Evaluation Report Generator

Orchestrates visualization generation from aggregated evaluation results.
Provides high-level interface for creating comprehensive evaluation reports.
"""

import os
import logging
from typing import Dict, Optional

from config import Config
from visualization import VisualizationFramework

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """
    High-level interface for generating evaluation reports and visualizations.
    
    Usage:
        generator = EvaluationReportGenerator()
        report_metadata = generator.generate_full_report()
    """
    
    def __init__(self, aggregated_summary_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            aggregated_summary_path: Path to aggregated_summary.json 
                                    (defaults to standard location)
            output_dir: Output directory for plots (defaults to Config.PLOTS_DIR)
        """
        # Default paths
        if aggregated_summary_path is None:
            aggregated_summary_path = os.path.join(
                Config.EVALUATIONS_DIR, 
                "aggregated_summary.json"
            )
        
        if output_dir is None:
            output_dir = Config.PLOTS_DIR
        
        self.aggregated_summary_path = aggregated_summary_path
        self.output_dir = output_dir
        
        # Validate input file exists
        if not os.path.exists(aggregated_summary_path):
            raise FileNotFoundError(
                f"Aggregated summary not found: {aggregated_summary_path}\n"
                f"Please run 'python main.py evaluate' first to generate aggregated results."
            )
        
        # Initialize visualization framework
        self.viz_framework = VisualizationFramework(
            aggregated_json_path=aggregated_summary_path,
            output_dir=output_dir
        )
        
        logger.info(f"EvaluationReportGenerator initialized")
        logger.info(f"  Input: {aggregated_summary_path}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Setups: {len(self.viz_framework.navigator.setups)}")
    
    def generate_full_report(self) -> Dict[str, any]:
        """
        Generate complete evaluation report with all visualizations.
        
        Returns:
            dict: Report metadata including:
                - output_directory: Where plots were saved
                - plots_generated: Setup -> list of plot files
                - html_dashboard: Path to HTML dashboard
                - total_plots: Total number of plots generated
        """
        logger.info("=" * 80)
        logger.info("GENERATING FULL EVALUATION REPORT")
        logger.info("=" * 80)
        
        # Generate all visualizations
        plots_by_setup = self.viz_framework.generate_all_visualizations()
        
        # Count total plots
        total_plots = sum(len(plots) for plots in plots_by_setup.values())
        
        # Dashboard path
        dashboard_path = os.path.join(self.output_dir, "dashboard.html")
        
        # Compile report metadata
        report_metadata = {
            'output_directory': self.output_dir,
            'plots_generated': plots_by_setup,
            'html_dashboard': dashboard_path,
            'total_plots': total_plots,
            'setups_processed': len(plots_by_setup),
            'aggregated_summary_source': self.aggregated_summary_path
        }
        
        # Log summary
        logger.info("=" * 80)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total plots generated: {total_plots}")
        logger.info(f"Setups processed: {len(plots_by_setup)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"HTML Dashboard: {dashboard_path}")
        logger.info("=" * 80)
        
        return report_metadata
    
    def generate_setup_report(self, setup_name: str) -> Dict[str, any]:
        """
        Generate report for a specific setup only.
        
        Args:
            setup_name: Name of the setup to visualize
            
        Returns:
            dict: Report metadata for this setup
        """
        if setup_name not in self.viz_framework.navigator.setups:
            raise ValueError(
                f"Setup '{setup_name}' not found in aggregated data. "
                f"Available setups: {self.viz_framework.navigator.setups}"
            )
        
        logger.info(f"Generating report for setup: {setup_name}")
        
        # Import here to avoid circular dependency
        from visualization import SetupPlotter
        
        plotter = SetupPlotter(
            setup_name=setup_name,
            navigator=self.viz_framework.navigator,
            output_dir=self.output_dir
        )
        
        plot_files = plotter.plot_all()
        
        report_metadata = {
            'setup_name': setup_name,
            'output_directory': self.output_dir,
            'plots_generated': plot_files,
            'total_plots': len(plot_files)
        }
        
        logger.info(f"Generated {len(plot_files)} plots for setup '{setup_name}'")
        
        return report_metadata