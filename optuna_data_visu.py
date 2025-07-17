import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OptunaAnalyzer:
    def __init__(self, unet_csv_path, segformer_csv_path):
        """
        Initialize the analyzer with paths to CSV files for both models
        """
        self.unet_data = pd.read_csv(unet_csv_path)
        self.segformer_data = pd.read_csv(segformer_csv_path)
        
        # Add model labels
        self.unet_data['model'] = 'UNet'
        self.segformer_data['model'] = 'SegFormer'
        
        # Combine data
        self.combined_data = pd.concat([self.unet_data, self.segformer_data], ignore_index=True)
        
        # Get unique trials by grouping by hyperparameters
        self.unet_trials = self._get_trial_summary(self.unet_data)
        self.segformer_trials = self._get_trial_summary(self.segformer_data)
        self.all_trials = pd.concat([self.unet_trials, self.segformer_trials], ignore_index=True)
        
    def _identify_trials(self, data):
        """Identify individual trials when epochs reset"""
        data = data.copy()
        data['trial_id'] = 0
        
        trial_id = 0
        for i in range(1, len(data)):
            # Check if epoch decreased (indicating new trial)
            if data.iloc[i]['epoch'] < data.iloc[i-1]['epoch']:
                trial_id += 1
            data.iloc[i, data.columns.get_loc('trial_id')] = trial_id
        
        return data
    
    def _get_trial_summary(self, data):
        """Extract trial-level summary statistics"""
        # First identify individual trials
        data_with_trials = self._identify_trials(data)
        
        # Group by trial_id and hyperparameters to get unique trials
        trial_groups = data_with_trials.groupby(['trial_id', 'bs', 'lr', 'weightdecay', 'accusteps'])
        
        trial_summaries = []
        for name, group in trial_groups:
            trial_id, bs, lr, weightdecay, accusteps = name
            
            # Get final epoch performance for this trial
            final_epoch = group.iloc[-1]
            
            # Calculate convergence metrics within this trial
            best_dice = group['dice'].max()
            best_iou = group['iou'].max()
            epochs_to_90_percent = self._epochs_to_threshold(group, 'dice', 0.9 * best_dice)
            
            # Check if early stopping occurred (epoch didn't reach expected max)
            early_stopped = group['epoch'].max() < 25  # Assuming 25 is your max epochs
            
            summary = {
                'trial_id': trial_id,
                'bs': bs,
                'lr': lr,
                'weightdecay': weightdecay,
                'accusteps': accusteps,
                'model': group['model'].iloc[0],
                'final_dice': final_epoch['dice'],
                'final_iou': final_epoch['iou'],
                'final_precision': final_epoch['precision'],
                'final_recall': final_epoch['recall'],
                'best_dice': best_dice,
                'best_iou': best_iou,
                'epochs_to_90_percent': epochs_to_90_percent,
                'total_epochs': len(group),
                'max_epoch_reached': group['epoch'].max(),
                'early_stopped': early_stopped,
                'final_val_loss': final_epoch['val_loss'],
                'final_train_loss': final_epoch['train_loss']
            }
            trial_summaries.append(summary)
        
        return pd.DataFrame(trial_summaries)
    
    def _epochs_to_threshold(self, group, metric, threshold):
        """Calculate epochs needed to reach threshold"""
        above_threshold = group[group[metric] >= threshold]
        return len(above_threshold) if len(above_threshold) > 0 else group.shape[0]
    
    def plot_training_curves(self, figsize=(15, 10), show_all_trials=False):
        """Plot training curves for both models"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        if show_all_trials:
            # Show multiple trials with transparency
            self._plot_all_trials(axes)
        else:
            # Show only best trials
            self._plot_best_trials(axes)
        
        plt.tight_layout()
        plt.show()
        
    def plot_convergence_analysis(self, figsize=(15, 8)):
        """Analyze convergence patterns and early stopping"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Early stopping analysis
        early_stop_counts = self.all_trials.groupby('model')['early_stopped'].value_counts().unstack(fill_value=0)
        early_stop_counts.plot(kind='bar', ax=axes[0,0], color=['lightgreen', 'lightcoral'])
        axes[0,0].set_title('Early Stopping Frequency')
        axes[0,0].set_xlabel('Model')
        axes[0,0].set_ylabel('Number of Trials')
        axes[0,0].legend(['Completed', 'Early Stopped'])
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # Epochs reached distribution
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model]
            axes[0,1].hist(model_data['total_epochs'], alpha=0.7, bins=15, 
                          label=model, color=['blue', 'red'][i])
        axes[0,1].set_title('Distribution of Training Epochs')
        axes[0,1].set_xlabel('Total Epochs')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Convergence speed (epochs to 90% best performance)
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model]
            valid_convergence = model_data[model_data['epochs_to_90_percent'] < model_data['total_epochs']]
            axes[0,2].hist(valid_convergence['epochs_to_90_percent'], alpha=0.7, bins=10, 
                          label=model, color=['blue', 'red'][i])
        axes[0,2].set_title('Convergence Speed (Epochs to 90% Best)')
        axes[0,2].set_xlabel('Epochs')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].legend()
        
        # Performance vs epochs completed
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model]
            axes[1,0].scatter(model_data['total_epochs'], model_data['final_dice'], 
                             alpha=0.7, s=50, label=model, color=['blue', 'red'][i])
        axes[1,0].set_title('Performance vs Training Duration')
        axes[1,0].set_xlabel('Total Epochs')
        axes[1,0].set_ylabel('Final Dice Score')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Early stopping vs performance
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model]
            early_stopped = model_data[model_data['early_stopped'] == True]
            completed = model_data[model_data['early_stopped'] == False]
            
            axes[1,1].scatter(early_stopped['max_epoch_reached'], early_stopped['final_dice'], 
                             alpha=0.7, s=50, marker='x', color=['blue', 'red'][i], 
                             label=f'{model} (Early Stopped)')
            axes[1,1].scatter(completed['max_epoch_reached'], completed['final_dice'], 
                             alpha=0.7, s=50, marker='o', color=['blue', 'red'][i], 
                             label=f'{model} (Completed)')
        axes[1,1].set_title('Early Stopping vs Performance')
        axes[1,1].set_xlabel('Max Epoch Reached')
        axes[1,1].set_ylabel('Final Dice Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Trial progression (for debugging)
        axes[1,2].text(0.1, 0.9, f"UNet Trials: {len(self.unet_trials)}", 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].text(0.1, 0.8, f"SegFormer Trials: {len(self.segformer_trials)}", 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].text(0.1, 0.7, f"Total Epochs (UNet): {len(self.unet_data)}", 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].text(0.1, 0.6, f"Total Epochs (SegFormer): {len(self.segformer_data)}", 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].set_title('Dataset Summary')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _plot_best_trials(self, axes):
        """Plot best trials for each model"""
        # Get best trial for each model based on dice score
        best_unet_trial = self.unet_trials.loc[self.unet_trials['final_dice'].idxmax()]
        best_segformer_trial = self.segformer_trials.loc[self.segformer_trials['final_dice'].idxmax()]
        
        # Get data with trial IDs
        unet_data_with_trials = self._identify_trials(self.unet_data)
        segformer_data_with_trials = self._identify_trials(self.segformer_data)
        
        # Filter data for best trials
        best_unet_data = unet_data_with_trials[
            unet_data_with_trials['trial_id'] == best_unet_trial['trial_id']
        ]
        
        best_segformer_data = segformer_data_with_trials[
            segformer_data_with_trials['trial_id'] == best_segformer_trial['trial_id']
        ]
        
        self._plot_metrics(axes, best_unet_data, best_segformer_data)

    def plot_partial_dependence_iou(self, param='lr', bins=10):
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model].copy()
            
            # Binning hiperparametra
            if param in ['lr', 'weightdecay']:
                model_data['param_bin'] = pd.qcut(model_data[param], bins, duplicates='drop')
            else:
                model_data['param_bin'] = model_data[param]
            
            grouped = model_data.groupby('param_bin')['final_iou'].agg(['mean', 'std']).reset_index()
            
            # Plot sa error barovima (mean ± std)
            axes[i].errorbar(grouped['param_bin'].astype(str), grouped['mean'], yerr=grouped['std'], fmt='-o')
            axes[i].set_title(f'{model} - Partial Dependence of Final IoU on {param}')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Final IoU')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


    def plot_heatmap_hyperparam_combinations(self, param1='lr', param2='weightdecay', metric='final_dice'):

        df = self.all_trials.copy()
        df[param1 + '_binned'] = np.log10(df[param1]).round(3)
        df[param2 + '_binned'] = np.log10(df[param2]).round(3)
        
        pivot_table = df.pivot_table(index=param1 + '_binned', columns=param2 + '_binned', values=metric, aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap='viridis')
        plt.title(f'Heatmap of mean {metric} by {param1} and {param2} (log10 binned)')
        plt.xlabel(f'{param2} (log10 binned)')
        plt.ylabel(f'{param1} (log10 binned)')
        plt.tight_layout()
        plt.show()


    def plot_partial_dependence(self, param='lr', metric='final_dice', bins=10):
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model].copy()
            
            # Binning parametra
            if param in ['lr', 'weightdecay']:
                model_data['param_bin'] = pd.qcut(model_data[param], bins, duplicates='drop')
            else:
                model_data['param_bin'] = model_data[param]
            
            grouped = model_data.groupby('param_bin')[metric].agg(['mean', 'std']).reset_index()
            
            # Plot with error bars
            axes[i].errorbar(grouped['param_bin'].astype(str), grouped['mean'], yerr=grouped['std'], fmt='-o')
            axes[i].set_title(f'{model} - Partial Dependence of {metric} on {param}')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _plot_all_trials(self, axes):
        """Plot all trials with transparency"""
        unet_data_with_trials = self._identify_trials(self.unet_data)
        segformer_data_with_trials = self._identify_trials(self.segformer_data)
        
        # Plot all UNet trials
        for trial_id in unet_data_with_trials['trial_id'].unique():
            trial_data = unet_data_with_trials[unet_data_with_trials['trial_id'] == trial_id]
            axes[0,0].plot(trial_data['epoch'], trial_data['val_loss'], 
                          color='blue', alpha=0.3, linewidth=1)
            axes[0,1].plot(trial_data['epoch'], trial_data['dice'], 
                          color='blue', alpha=0.3, linewidth=1)
            axes[1,0].plot(trial_data['epoch'], trial_data['iou'], 
                          color='blue', alpha=0.3, linewidth=1)
        
        # Plot all SegFormer trials
        for trial_id in segformer_data_with_trials['trial_id'].unique():
            trial_data = segformer_data_with_trials[segformer_data_with_trials['trial_id'] == trial_id]
            axes[0,0].plot(trial_data['epoch'], trial_data['val_loss'], 
                          color='red', alpha=0.3, linewidth=1)
            axes[0,1].plot(trial_data['epoch'], trial_data['dice'], 
                          color='red', alpha=0.3, linewidth=1)
            axes[1,0].plot(trial_data['epoch'], trial_data['iou'], 
                          color='red', alpha=0.3, linewidth=1)
        
        # Add titles and labels
        axes[0,0].set_title('All Trials - Validation Loss')
        axes[0,1].set_title('All Trials - Dice Score')
        axes[1,0].set_title('All Trials - IoU')
        axes[1,1].set_title('Trial Distribution')
        
        # Add legend
        axes[0,0].plot([], [], color='blue', alpha=0.7, label='UNet')
        axes[0,0].plot([], [], color='red', alpha=0.7, label='SegFormer')
        axes[0,0].legend()
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
    
    def _plot_metrics(self, axes, unet_data, segformer_data):
        """Plot metrics for specific trials"""
        # Plot loss curves
        axes[0,0].plot(unet_data['epoch'], unet_data['train_loss'], 
                       label='UNet Train', color='blue', alpha=0.7)
        axes[0,0].plot(unet_data['epoch'], unet_data['val_loss'], 
                       label='UNet Val', color='blue', linestyle='--')
        axes[0,0].plot(segformer_data['epoch'], segformer_data['train_loss'], 
                       label='SegFormer Train', color='red', alpha=0.7)
        axes[0,0].plot(segformer_data['epoch'], segformer_data['val_loss'], 
                       label='SegFormer Val', color='red', linestyle='--')
        axes[0,0].set_title('Training and Validation Loss (Best Trials)')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot Dice score
        axes[0,1].plot(unet_data['epoch'], unet_data['dice'], 
                       label='UNet', color='blue', marker='o', markersize=3)
        axes[0,1].plot(segformer_data['epoch'], segformer_data['dice'], 
                       label='SegFormer', color='red', marker='s', markersize=3)
        axes[0,1].set_title('Dice Score Evolution (Best Trials)')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Dice Score')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot IoU
        axes[1,0].plot(unet_data['epoch'], unet_data['iou'], 
                       label='UNet', color='blue', marker='o', markersize=3)
        axes[1,0].plot(segformer_data['epoch'], segformer_data['iou'], 
                       label='SegFormer', color='red', marker='s', markersize=3)
        axes[1,0].set_title('IoU Evolution (Best Trials)')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('IoU')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot Precision vs Recall
        axes[1,1].plot(unet_data['epoch'], unet_data['precision'], 
                       label='UNet Precision', color='blue', alpha=0.7)
        axes[1,1].plot(unet_data['epoch'], unet_data['recall'], 
                       label='UNet Recall', color='blue', linestyle='--')
        axes[1,1].plot(segformer_data['epoch'], segformer_data['precision'], 
                       label='SegFormer Precision', color='red', alpha=0.7)
        axes[1,1].plot(segformer_data['epoch'], segformer_data['recall'], 
                       label='SegFormer Recall', color='red', linestyle='--')
        axes[1,1].set_title('Precision and Recall Evolution (Best Trials)')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    def plot_performance_comparison(self, figsize=(12, 8)):
        """Create box plots comparing model performance"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        metrics = ['final_dice', 'final_iou', 'final_precision', 'final_recall']
        titles = ['Dice Score', 'IoU', 'Precision', 'Recall']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            
            # Create box plot
            data_to_plot = [
                self.unet_trials[metric].values,
                self.segformer_trials[metric].values
            ]
            
            bp = axes[row, col].boxplot(data_to_plot, labels=['UNet', 'SegFormer'], 
                                       patch_artist=True, notch=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[row, col].set_title(f'{title} Distribution')
            axes[row, col].set_ylabel(title)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add statistical annotation
            unet_scores = self.unet_trials[metric].values
            segformer_scores = self.segformer_trials[metric].values
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(unet_scores, segformer_scores)
            
            # Add p-value annotation
            axes[row, col].text(0.02, 0.98, f'p = {p_value:.4f}', 
                               transform=axes[row, col].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_hyperparameter_analysis(self, figsize=(15, 12)):
        """Analyze hyperparameter sensitivity"""
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Learning rate vs performance
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model]
            
            # LR vs Dice
            axes[0, i].scatter(model_data['lr'], model_data['final_dice'], 
                              alpha=0.7, s=50, c=model_data['final_dice'], 
                              cmap='viridis')
            axes[0, i].set_xlabel('Learning Rate')
            axes[0, i].set_ylabel('Final Dice Score')
            axes[0, i].set_title(f'{model} - LR vs Dice')
            axes[0, i].set_xscale('log')
            axes[0, i].grid(True, alpha=0.3)
            
            # Batch size vs performance
            axes[1, i].scatter(model_data['bs'], model_data['final_dice'], 
                              alpha=0.7, s=50, c=model_data['final_dice'], 
                              cmap='viridis')
            axes[1, i].set_xlabel('Batch Size')
            axes[1, i].set_ylabel('Final Dice Score')
            axes[1, i].set_title(f'{model} - Batch Size vs Dice')
            axes[1, i].grid(True, alpha=0.3)
            
            # Weight decay vs performance
            axes[2, i].scatter(model_data['weightdecay'], model_data['final_dice'], 
                              alpha=0.7, s=50, c=model_data['final_dice'], 
                              cmap='viridis')
            axes[2, i].set_xlabel('Weight Decay')
            axes[2, i].set_ylabel('Final Dice Score')
            axes[2, i].set_title(f'{model} - Weight Decay vs Dice')
            axes[2, i].set_xscale('log')
            axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_correlation_heatmap(self, figsize=(12, 8)):
        """Create correlation heatmap between hyperparameters and performance"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Prepare data for correlation
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model]
            
            # Select relevant columns
            corr_data = model_data[['lr', 'bs', 'weightdecay', 'accusteps', 
                                   'final_dice', 'final_iou', 'final_precision', 
                                   'final_recall']].copy()
            
            # Calculate correlation matrix
            corr_matrix = corr_data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[i], square=True, linewidths=0.5)
            axes[i].set_title(f'{model} - Hyperparameter Correlation')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_trial_variability(self, metric='final_dice', figsize=(10, 6)):
        """Vizualizuj raspodelu metrika po trialovima"""
        plt.figure(figsize=figsize)
        sns.boxplot(data=self.all_trials, x='model', y=metric, palette=['skyblue', 'lightcoral'])
        sns.stripplot(data=self.all_trials, x='model', y=metric, color='black', alpha=0.5, jitter=True)
        plt.title(f'Trial Variability for {metric.upper()}')
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_mean_std_curves(self, metric='dice', figsize=(10, 6)):
        """Prikaz prosečne vrednosti i standardne devijacije metrika po epohama"""
        fig, ax = plt.subplots(figsize=figsize)
        for model_data, label, color in zip(
            [self.unet_data, self.segformer_data],
            ['UNet', 'SegFormer'],
            ['blue', 'red']
        ):
            df = self._identify_trials(model_data)
            grouped = df.groupby(['epoch'])[metric]
            mean = grouped.mean()
            std = grouped.std()

            ax.plot(mean.index, mean.values, label=f'{label} Mean', color=color)
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_title(f'Mean ± Std of {metric.title()} Across Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_radar_performance(self):
        """Radar graf prosečnih performansi po modelu"""
        from math import pi

        categories = ['final_dice', 'final_iou', 'final_precision', 'final_recall']
        categories_display = ['Dice', 'IoU', 'Precision', 'Recall']

        df = self.all_trials.groupby('model')[categories].mean().reset_index()

        df_norm = df.copy()
        df_norm[categories] = df[categories] / df[categories].max()

        labels = df_norm['model'].values
        stats_ = df_norm[categories].values

        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for i, stat in enumerate(stats_):
            values = list(stat)
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=labels[i])
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_display)
        ax.set_title('Aggregated Performance (Normalized)', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()

        
    def plot_violin_metric_variability(self, param='lr', metric='final_dice', bins=6, figsize=(12, 6)):
        """
        Prikazuje violin plot raspodele metrike (npr. final_dice) po binovima vrednosti
        odabranog hiperparametra (npr. lr).
        """
        import matplotlib.ticker as ticker

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for i, model in enumerate(['UNet', 'SegFormer']):
            model_data = self.all_trials[self.all_trials['model'] == model].copy()
            
            # Binarizacija hiperparametra za kontinuirane vrednosti
            if param in ['lr', 'weightdecay']:
                model_data['param_bin'] = pd.qcut(model_data[param], bins, duplicates='drop')
            else:
                model_data['param_bin'] = model_data[param]
            
            sns.violinplot(x='param_bin', y=metric, data=model_data, ax=axes[i], palette='muted')
            axes[i].set_title(f'{model} - Variability of {metric} across {param} bins')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)

            # Poboljšanje prikaza x-osa ako su kategorije intervali
            if param in ['lr', 'weightdecay']:
                axes[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=bins))
            
        plt.tight_layout()
        plt.show()

    def plot_performance_comparison(self, figsize=(12, 8)):
        """Create box plots comparing model performance with annotations"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        metrics = ['final_dice', 'final_iou', 'final_precision', 'final_recall']
        titles = ['Dice Score', 'IoU', 'Precision', 'Recall']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2

            data_to_plot = [
                self.unet_trials[metric].values,
                self.segformer_trials[metric].values
            ]

            bp = axes[row, col].boxplot(data_to_plot, labels=['UNet', 'SegFormer'],
                                        patch_artist=True, notch=True)

            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            axes[row, col].set_title(f'{title} Distribution')
            axes[row, col].set_ylabel(title)
            axes[row, col].grid(True, alpha=0.3)

            unet_scores = self.unet_trials[metric].values
            segformer_scores = self.segformer_trials[metric].values

            t_stat, p_value = stats.ttest_ind(unet_scores, segformer_scores)
            delta = np.mean(unet_scores) - np.mean(segformer_scores)

            annotation = f'p = {p_value:.4f}\nΔ = {delta:.4f}'
            axes[row, col].text(0.02, 0.98, annotation,
                                transform=axes[row, col].transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

        return fig

    
    def generate_performance_table(self):
        """Generate summary statistics table"""
        summary_stats = []
        
        for model in ['UNet', 'SegFormer']:
            model_data = self.all_trials[self.all_trials['model'] == model]
            
            stats_dict = {
                'Model': model,
                'Dice (Mean±SD)': f"{model_data['final_dice'].mean():.4f}±{model_data['final_dice'].std():.4f}",
                'IoU (Mean±SD)': f"{model_data['final_iou'].mean():.4f}±{model_data['final_iou'].std():.4f}",
                'Precision (Mean±SD)': f"{model_data['final_precision'].mean():.4f}±{model_data['final_precision'].std():.4f}",
                'Recall (Mean±SD)': f"{model_data['final_recall'].mean():.4f}±{model_data['final_recall'].std():.4f}",
                'Best Dice': f"{model_data['final_dice'].max():.4f}",
                'Best IoU': f"{model_data['final_iou'].max():.4f}",
                'Trials': len(model_data)
            }
            summary_stats.append(stats_dict)
        
        return pd.DataFrame(summary_stats)
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests"""
        unet_dice = self.unet_trials['final_dice'].values
        segformer_dice = self.segformer_trials['final_dice'].values
        
        # T-test
        t_stat, t_p = stats.ttest_ind(unet_dice, segformer_dice)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(unet_dice, segformer_dice, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(unet_dice) - 1) * np.var(unet_dice, ddof=1) + 
                             (len(segformer_dice) - 1) * np.var(segformer_dice, ddof=1)) / 
                            (len(unet_dice) + len(segformer_dice) - 2))
        cohens_d = (np.mean(unet_dice) - np.mean(segformer_dice)) / pooled_std
        
        results = {
            'T-test': {'statistic': t_stat, 'p-value': t_p},
            'Mann-Whitney U': {'statistic': u_stat, 'p-value': u_p},
            'Effect Size (Cohen\'s d)': cohens_d,
            'Interpretation': 'Small effect' if abs(cohens_d) < 0.5 else 'Medium effect' if abs(cohens_d) < 0.8 else 'Large effect'
        }
        
        return results
    
    def get_best_hyperparameters(self):
        """Get best hyperparameter configurations"""
        best_configs = {}
        
        for model in ['UNet', 'SegFormer']:
            model_data = self.all_trials[self.all_trials['model'] == model]
            best_trial = model_data.loc[model_data['final_dice'].idxmax()]
            
            best_configs[model] = {
                'Learning Rate': best_trial['lr'],
                'Batch Size': best_trial['bs'],
                'Weight Decay': best_trial['weightdecay'],
                'Accumulation Steps': best_trial['accusteps'],
                'Final Dice': best_trial['final_dice'],
                'Final IoU': best_trial['final_iou']
            }
        
        return best_configs

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = OptunaAnalyzer('epoch_metrics_unet_new.csv', 'epoch_metrics_seg_new.csv')
    
    # Generate all visualizations
    print("Generating training curves...")
    analyzer.plot_training_curves()
    
    print("Generating training curves for all trials...")
    analyzer.plot_training_curves(show_all_trials=True)
    
    print("Generating convergence analysis...")
    analyzer.plot_convergence_analysis()
    
    print("Generating performance comparison...")
    analyzer.plot_performance_comparison()
    
    print("Generating hyperparameter analysis...")
    analyzer.plot_hyperparameter_analysis()
    
    print("Generating correlation heatmap...")
    analyzer.plot_correlation_heatmap()

     # Partial dependence za IOU po learning rate
    analyzer.plot_partial_dependence_iou(param='lr', bins=10)

    # Heatmap kombinacija dva hiperparametra
    analyzer.plot_heatmap_hyperparam_combinations(param1='lr', param2='weightdecay', metric='final_dice')

    # General partial dependence za batch size i preciznost
    analyzer.plot_partial_dependence(param='bs', metric='final_precision', bins=5)

    # Violin plot varijabilnosti final_dice po learning rate binovima
    analyzer.plot_violin_metric_variability(param='lr', metric='final_dice', bins=6)

    # Generate summary table
    print("\nPerformance Summary:")
    print(analyzer.generate_performance_table().to_string(index=False))
    
    # Statistical tests
    print("\nStatistical Tests:")
    stats_results = analyzer.perform_statistical_tests()
    for test, result in stats_results.items():
        if isinstance(result, dict):
            print(f"{test}: statistic={result['statistic']:.4f}, p-value={result['p-value']:.4f}")
        else:
            print(f"{test}: {result}")
    
    # Best hyperparameters
    print("\nBest Hyperparameters:")
    best_configs = analyzer.get_best_hyperparameters()
    for model, config in best_configs.items():
        print(f"\n{model}:")
        for param, value in config.items():
            print(f"  {param}: {value}")