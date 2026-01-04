"""
Daily AI Assistant Usage Behavior Analysis
Command-line version based on Kaggle notebook structure
"""

from analysis import AIUsageBehaviorAnalysis
import sys


def main():
    """Main execution function"""
    print("="*70)
    print("AI ASSISTANT USAGE BEHAVIOR ANALYSIS")
    print("="*70)
    
    # Check if dataset exists
    dataset_path = 'Daily_AI_Assistant_Usage_Behavior_Dataset.csv'
    
    # Initialize and run analysis
    analysis = AIUsageBehaviorAnalysis(dataset_path)
    analysis.run_full_analysis()
    
    # Example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    # Example 1
    print("\nExample 1: Desktop Coding Session")
    analysis.predict(
        device='Desktop',
        usage_category='Coding',
        assistant_model='GPT-5',
        prompt_length=150,
        hour=14,
        day_of_week=2,
        month=1
    )
    
    # Example 2
    print("\nExample 2: Mobile Research Session")
    analysis.predict(
        device='Mobile',
        usage_category='Research',
        assistant_model='o1',
        prompt_length=80,
        hour=20,
        day_of_week=5,
        month=1
    )
    
    # Example 3
    print("\nExample 3: Tablet Writing Session")
    analysis.predict(
        device='Tablet',
        usage_category='Writing',
        assistant_model='GPT-4o',
        prompt_length=200,
        hour=10,
        day_of_week=1,
        month=1
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - Check generated PNG files for visualizations")
    print("="*70)


if __name__ == '__main__':
    main()

