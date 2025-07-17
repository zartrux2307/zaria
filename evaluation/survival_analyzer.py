"""
Advanced Survival Analysis for Nonce Quality Evaluation

Features:
- Real-world duration modeling based on blockchain metrics
- Covariate analysis of nonce quality parameters
- Hazard ratio interpretation for mining efficiency
- Integration with project configuration
- Production-ready error handling and logging

Usage:
    analyzer = SurvivalAnalyzer()
    results = analyzer.analyze_survival()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import logging
import sys
from datetime import datetime
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from iazar.utils.nonce_loader import NonceLoader
from iazar.utils.config_manager import ConfigManager
from iazar.generator.config_loader import config_loader
# Project setup
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/survival_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SurvivalAnalyzer")

class SurvivalAnalyzer:
    def __init__(self, config=None):
        self.config_manager = ConfigManager()
        self.project_config = self.config_manager.get_config('global_config')
        self.ia_config = self.project_config.get('ia', {})
        
        self.loader = NonceLoader(config=config)
        self.data_path = os.path.join(self.loader.training_dir, "nonce_training_data.csv")
        self.results_dir = os.path.join(self.loader.data_dir, "survival_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configuration parameters
        self.analysis_params = self.ia_config.get('survival_analysis', {
            'min_sample_size': 1000,
            'max_correlation': 0.7,
            'significant_p_value': 0.05
        })
        
        # Core analysis columns
        self.REQUIRED_COLUMNS = [
            'nonce', 'entropy', 'uniqueness', 'zero_density', 
            'pattern_score', 'is_valid', 'block_height'
        ]
        self.DURATION_COL = 'mining_duration'
        self.EVENT_COL = 'acceptance_event'

    def _load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess data with industrial-grade validation"""
        logger.info(f"Loading mining data from: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            logger.error("Data file not found")
            return pd.DataFrame()

        try:
            # Load with strict validation
            df = pd.read_csv(
                self.data_path, 
                usecols=self.REQUIRED_COLUMNS,
                dtype={'nonce': 'string', 'block_height': 'int32'},
                on_bad_lines='skip'
            )
            
            # Validate critical columns
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                logger.error(f"Critical columns missing: {', '.join(missing_cols)}")
                return pd.DataFrame()
                
            # Create duration metric (blocks until acceptance)
            df[self.DURATION_COL] = df.groupby('block_height')['block_height'].transform(
                lambda x: x - x.min() + 1
            )
            
            # Create event metric (1 = accepted, 0 = rejected)
            df[self.EVENT_COL] = df['is_valid'].astype(int)
            
            # Quality control metrics
            df = df.dropna(subset=self.REQUIRED_COLUMNS)
            df = df[df['entropy'] > 0]  # Filter invalid entropy
            
            logger.info(f"Loaded {len(df)} validated mining records")
            return df

        except Exception as e:
            logger.exception(f"Data loading failed: {str(e)}")
            return pd.DataFrame()

    def _check_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate dataset meets analysis requirements"""
        if len(df) < self.analysis_params['min_sample_size']:
            logger.error(f"Insufficient data: {len(df)} records < minimum {self.analysis_params['min_sample_size']}")
            return False
            
        if df[self.EVENT_COL].sum() < 50:
            logger.error("Insufficient acceptance events for analysis")
            return False
            
        # Check for multicollinearity
        corr_matrix = df[['entropy', 'uniqueness', 'zero_density', 'pattern_score']].corr().abs()
        if (corr_matrix > self.analysis_params['max_correlation']).sum().sum() > 4:
            logger.warning("High multicollinearity detected in quality metrics")
            
        return True

    def perform_kaplan_meier(self, df: pd.DataFrame) -> KaplanMeierFitter:
        """Professional Kaplan-Meier estimation with confidence intervals"""
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df[self.DURATION_COL],
                event_observed=df[self.EVENT_COL],
                label="Nonce Acceptance"
            )
            
            # Statistical validation
            groups = df['entropy'] > df['entropy'].median()
            high_ent = df[groups][self.DURATION_COL]
            low_ent = df[~groups][self.DURATION_COL]
            high_event = df[groups][self.EVENT_COL]
            low_event = df[~groups][self.EVENT_COL]
            
            results = logrank_test(high_ent, low_ent, high_event, low_event)
            logger.info(f"KM Log-Rank Test: p={results.p_value:.4f}")
            
            return kmf
        except Exception as e:
            logger.exception(f"Kaplan-Meier failed: {str(e)}")
            return None

    def perform_cox_regression(self, df: pd.DataFrame, covariates: list) -> tuple:
        """Industrial-strength Cox proportional hazards modeling"""
        try:
            analysis_df = df[[self.DURATION_COL, self.EVENT_COL] + covariates].dropna()
            
            if len(analysis_df) < self.analysis_params['min_sample_size']:
                logger.error("Insufficient data for Cox PH model")
                return None, 0.0
                
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(
                analysis_df, 
                duration_col=self.DURATION_COL,
                event_col=self.EVENT_COL,
                show_progress=False
            )
            
            # Model diagnostics
            concordance = concordance_index(
                analysis_df[self.DURATION_COL],
                -cph.predict_partial_hazard(analysis_df[covariates]),
                analysis_df[self.EVENT_COL]
            )
            logger.info(f"Cox PH Concordance Index: {concordance:.4f}")
            
            # Validate proportional hazards assumption
            cph.check_assumptions(analysis_df, show_plots=False)
            
            return cph, concordance
        except Exception as e:
            logger.exception(f"Cox PH modeling failed: {str(e)}")
            return None, 0.0

    def visualize_survival(self, kmf: KaplanMeierFitter, title: str) -> str:
        """Generate publication-quality survival plots"""
        try:
            plt.figure(figsize=(12, 8))
            kmf.plot(ci_show=True, linewidth=2.5)
            plt.title(title, fontsize=16)
            plt.xlabel("Blocks Since Generation", fontsize=14)
            plt.ylabel("Acceptance Probability", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f"survival_curve_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
        except Exception as e:
            logger.error(f"Plot generation failed: {str(e)}")
            return ""

    def visualize_hazard_ratios(self, cph: CoxPHFitter) -> str:
        """Generate hazard ratio plot for key predictors"""
        try:
            plt.figure(figsize=(10, 6))
            cph.plot(hazard_ratios=True, figsize=(10, 6))
            plt.title("Quality Metric Hazard Ratios", fontsize=14)
            plt.xlabel("Hazard Ratio (log scale)")
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f"hazard_ratios_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
        except Exception as e:
            logger.error(f"Hazard ratio plot failed: {str(e)}")
            return ""

    def generate_report(self, results: dict) -> str:
        """Create comprehensive technical report"""
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "data_source": self.data_path,
            "records_processed": results.get("record_count", 0),
            "covariates_used": results.get("covariates", []),
            "kaplan_meier": {
                "median_survival": results.get("median_survival", 0),
                "mean_survival": results.get("mean_survival", 0)
            },
            "cox_model": {
                "concordance": results.get("cox_concordance", 0),
                "significant_predictors": results.get("significant_predictors", [])
            },
            "findings": results.get("key_findings", ""),
            "plots": {
                "survival_curve": results.get("survival_plot", ""),
                "hazard_ratios": results.get("hazard_plot", "")
            }
        }
        
        report_path = os.path.join(self.results_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report_path

    def analyze_survival(self, covariates=None):
        """Professional-grade survival analysis pipeline"""
        logger.info("===== INITIATING NONCE SURVIVAL ANALYSIS =====")
        
        # Phase 1: Data preparation
        df = self._load_and_preprocess()
        if df.empty or not self._check_data_quality(df):
            logger.error("Analysis aborted due to data issues")
            return None
            
        # Phase 2: Determine covariates
        if covariates is None:
            covariates = ['entropy', 'uniqueness', 'zero_density', 'pattern_score']
            logger.info(f"Using default covariates: {covariates}")
        
        # Phase 3: Kaplan-Meier analysis
        logger.info("Running Kaplan-Meier estimation...")
        kmf = self.perform_kaplan_meier(df)
        survival_plot = self.visualize_survival(kmf, "Nonce Acceptance Survival Curve")
        
        # Phase 4: Cox Regression
        logger.info("Running Cox Proportional Hazards modeling...")
        cph, concordance = self.perform_cox_regression(df, covariates)
        hazard_plot = self.visualize_hazard_ratios(cph) if cph else ""
        
        # Phase 5: Results compilation
        results = {
            "record_count": len(df),
            "covariates": covariates,
            "median_survival": kmf.median_survival_time_ if kmf else 0,
            "mean_survival": kmf.mean_survival_time_ if kmf else 0,
            "cox_concordance": concordance,
            "survival_plot": survival_plot,
            "hazard_plot": hazard_plot,
            "significant_predictors": [],
            "key_findings": ""
        }
        
        # Extract significant predictors
        if cph:
            significant = cph.summary[cph.summary]['p'] < self.analysis_params['significant_p_value']
            results["significant_predictors"] = significant.index.tolist()
            
            # Generate insights
            findings = []
            if 'entropy' in significant.index:
                hr = cph.hazard_ratios_['entropy']
                findings.append(f"Entropy significantly impacts acceptance (HR={hr:.2f})")
            if 'pattern_score' in significant.index:
                hr = cph.hazard_ratios_['pattern_score']
                findings.append(f"Pattern scores reduce acceptance probability (HR={hr:.2f})")
                
            results["key_findings"] = "; ".join(findings) if findings else "No significant predictors found"
        
        # Phase 6: Final reporting
        report_path = self.generate_report(results)
        logger.info(f"Analysis complete: report saved to {report_path}")
        logger.info("="*60)
        
        return results

def main():
    try:
        analyzer = SurvivalAnalyzer()
        results = analyzer.analyze_survival()
        return 0 if results else 1
    except Exception as e:
        logger.exception(f"Analysis failed: {str(e)}")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)