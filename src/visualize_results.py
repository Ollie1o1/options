#!/usr/bin/env python3
"""
Visualization module for options screener results.
Generates charts for IV analysis, risk/reward, and expected moves.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional


def create_visualizations(df: pd.DataFrame, mode: str, output_dir: str = "reports") -> None:
    """
    Generate visualization charts for options analysis.
    
    Args:
        df: DataFrame with options data including IV, HV, risk/reward metrics
        mode: Screening mode (e.g., "Single-stock", "Budget scan", "Discovery scan")
        output_dir: Directory to save charts
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Filter valid data
        required_cols = ["impliedVolatility", "hv_30d", "rr_ratio", "expected_move"]
        
        # Check for missing columns before dropping
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"  ⚠️  Missing columns for visualization: {missing_cols}")
            return

        df_clean = df.dropna(subset=required_cols)
        
        if df_clean.empty:
            print("  ⚠️  Insufficient data for visualization after filtering out NaNs.")
            return
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Options Analysis - {mode}", fontsize=16, fontweight="bold")
        
        # 1. IV vs HV Scatter Plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            df_clean["hv_30d"] * 100,
            df_clean["impliedVolatility"] * 100,
            c=df_clean["quality_score"],
            cmap="viridis",
            alpha=0.6,
            s=50
        )
        ax1.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=1, label="IV = HV")
        ax1.set_xlabel("Historical Volatility (30d) %", fontsize=10)
        ax1.set_ylabel("Implied Volatility %", fontsize=10)
        ax1.set_title("IV vs HV (colored by Quality Score)", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label="Quality Score")
        
        # 2. Risk/Reward Histogram
        ax2 = axes[0, 1]
        rr_data = df_clean["rr_ratio"].clip(lower=0, upper=5)  # Cap at 5:1 for visualization
        ax2.hist(rr_data, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
        ax2.axvline(rr_data.median(), color="red", linestyle="--", linewidth=2, label=f"Median: {rr_data.median():.2f}")
        ax2.set_xlabel("Risk/Reward Ratio", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)
        ax2.set_title("Risk/Reward Distribution", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Expected Move by Expiration
        ax3 = axes[1, 0]
        # Group by expiration and calculate average expected move
        df_clean["exp_date"] = pd.to_datetime(df_clean["expiration"])
        # Use T_years * 365 for DTE
        df_clean["dte"] = (df_clean["T_years"] * 365).astype(int)
        exp_grouped = df_clean.groupby("exp_date").agg({
            "expected_move": "mean",
            "underlying": "first",
            "dte": "first"
        }).sort_index()
        
        if not exp_grouped.empty:
            exp_grouped["exp_move_pct"] = (exp_grouped["expected_move"] / exp_grouped["underlying"]) * 100
            ax3.plot(exp_grouped.index, exp_grouped["exp_move_pct"], marker="o", linewidth=2, markersize=6, color="green")
            ax3.set_xlabel("Expiration Date", fontsize=10)
            ax3.set_ylabel("Expected Move %", fontsize=10)
            ax3.set_title("Expected Move by Expiration", fontsize=11, fontweight="bold")
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Quality Score Distribution by Bucket
        ax4 = axes[1, 1]
        if "price_bucket" in df_clean.columns:
            buckets = ["LOW", "MEDIUM", "HIGH"]
            bucket_data = [df_clean[df_clean["price_bucket"] == b]["quality_score"].dropna() for b in buckets]
            bucket_data = [d for d in bucket_data if len(d) > 0]  # Filter empty buckets
            bucket_labels = [b for b, d in zip(buckets, bucket_data) if len(d) > 0]
            
            if bucket_data:
                bp = ax4.boxplot(bucket_data, labels=bucket_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
                    patch.set_facecolor(color)
                ax4.set_xlabel("Premium Bucket", fontsize=10)
                ax4.set_ylabel("Quality Score", fontsize=10)
                ax4.set_title("Quality Score by Premium Bucket", fontsize=11, fontweight="bold")
                ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{output_dir}/options_analysis_{mode.replace(' ', '_')}_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  📊 Visualizations saved to: {filename}")
        
    except Exception as e:
        print(f"  ⚠️  Could not generate visualizations: {e}")


def create_backtest_charts(
    metrics_correlation: pd.DataFrame,
    performance_summary: pd.DataFrame,
    output_dir: str = "reports"
) -> None:
    """
    Generate charts for backtest results.
    
    Args:
        metrics_correlation: DataFrame with correlation between metrics and realized returns
        performance_summary: DataFrame with performance statistics
        output_dir: Directory to save charts
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Backtest Analysis", fontsize=16, fontweight="bold")
        
        # 1. Metric Correlations
        ax1 = axes[0]
        if not metrics_correlation.empty and "correlation" in metrics_correlation.columns:
            metrics_correlation = metrics_correlation.sort_values("correlation", ascending=True)
            colors = ['red' if x < 0 else 'green' for x in metrics_correlation["correlation"]]
            ax1.barh(metrics_correlation.index, metrics_correlation["correlation"], color=colors, alpha=0.7)
            ax1.set_xlabel("Correlation with Realized P/L", fontsize=10)
            ax1.set_title("Metric Correlations", fontsize=11, fontweight="bold")
            ax1.axvline(0, color='black', linewidth=0.8)
            ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Performance Summary
        ax2 = axes[1]
        if not performance_summary.empty:
            ax2.axis('tight')
            ax2.axis('off')
            table_data = [[str(v) for v in row] for row in performance_summary.values]
            table = ax2.table(
                cellText=table_data,
                colLabels=performance_summary.columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            ax2.set_title("Performance Summary", fontsize=11, fontweight="bold", pad=20)
        
        plt.tight_layout()
        
        filename = f"{output_dir}/backtest_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Backtest charts saved to: {filename}")
        
    except Exception as e:
        print(f"  ⚠️  Could not generate backtest charts: {e}")