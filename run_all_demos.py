#!/usr/bin/env python3
"""
CatBoost Project Demo Runner

This script runs all the demonstration scripts in the project to showcase
CatBoost's advantages and capabilities.
"""

import os
import sys
import subprocess
import time


def print_header(title):
    """Print a formatted header for each demo section."""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)


def run_script(script_path, description):
    """
    Run a Python script and handle any errors.
    
    Args:
        script_path (str): Path to the Python script
        description (str): Description of what the script does
    """
    print(f"\n🚀 Running: {description}")
    print(f"📁 Script: {script_path}")
    print("-" * 60)
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"✅ Successfully completed: {description}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}")
        print(f"Exit code: {e.returncode}")
        return False
        
    except FileNotFoundError:
        print(f"❌ Script not found: {script_path}")
        return False
    
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'catboost', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """Main function to run all demonstrations."""
    print_header("CATBOOST PROJECT DEMONSTRATION SUITE")
    
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies before running demos.")
        return
    
    print("✅ All dependencies are available!")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define demonstrations to run
    demonstrations = [
        {
            'script': os.path.join(script_dir, 'catboost_classifier.py'),
            'description': 'Basic CatBoost Implementation and Overview',
            'duration': '~2 minutes'
        },
        {
            'script': os.path.join(script_dir, 'examples', 'categorical_features_demo.py'),
            'description': 'Categorical Features Handling Demonstration',
            'duration': '~3 minutes'
        },
        {
            'script': os.path.join(script_dir, 'examples', 'overfitting_resistance_demo.py'),
            'description': 'Overfitting Resistance Analysis',
            'duration': '~4 minutes'
        },
        {
            'script': os.path.join(script_dir, 'model_comparison.py'),
            'description': 'Comprehensive Model Comparison',
            'duration': '~5 minutes'
        }
    ]
    
    print(f"\n📋 Demo Schedule ({len(demonstrations)} demonstrations):")
    total_duration = 0
    for i, demo in enumerate(demonstrations, 1):
        duration_minutes = int(demo['duration'].split('~')[1].split(' ')[0])
        total_duration += duration_minutes
        print(f"   {i}. {demo['description']} ({demo['duration']})")
    
    print(f"\n⏱️  Total estimated time: ~{total_duration} minutes")
    
    # Ask for confirmation
    response = input("\n🤔 Would you like to run all demonstrations? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("👋 Demo cancelled. You can run individual scripts manually:")
        for demo in demonstrations:
            print(f"   python {demo['script']}")
        return
    
    # Run each demonstration
    successful_runs = 0
    failed_runs = 0
    
    start_time = time.time()
    
    for i, demo in enumerate(demonstrations, 1):
        print_header(f"DEMO {i}/{len(demonstrations)}: {demo['description'].upper()}")
        
        success = run_script(demo['script'], demo['description'])
        
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
        
        # Add delay between demos for readability
        if i < len(demonstrations):
            print(f"\n⏳ Preparing next demonstration...")
            time.sleep(2)
    
    # Final summary
    total_time = time.time() - start_time
    
    print_header("DEMONSTRATION SUMMARY")
    
    print(f"📊 Results:")
    print(f"   ✅ Successful demonstrations: {successful_runs}")
    print(f"   ❌ Failed demonstrations: {failed_runs}")
    print(f"   ⏱️  Total execution time: {total_time/60:.1f} minutes")
    
    if successful_runs == len(demonstrations):
        print(f"\n🎉 All demonstrations completed successfully!")
        print(f"📚 For detailed analysis, check the generated documentation:")
        print(f"   - README.md: Project overview and usage")
        print(f"   - CATBOOST_ADVANTAGES.md: Technical advantages analysis")
    else:
        print(f"\n⚠️  Some demonstrations failed. Check the output above for details.")
    
    print(f"\n🔗 Key CatBoost Advantages Demonstrated:")
    advantages = [
        "✓ Native categorical feature handling",
        "✓ Superior overfitting resistance", 
        "✓ Minimal preprocessing requirements",
        "✓ Robust default parameters",
        "✓ Excellent small dataset performance",
        "✓ Built-in interpretability tools",
        "✓ High development efficiency"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")
    
    print(f"\n📖 Next Steps:")
    print(f"   1. Review the generated plots and analysis")
    print(f"   2. Read CATBOOST_ADVANTAGES.md for technical details")
    print(f"   3. Experiment with your own datasets")
    print(f"   4. Integrate CatBoost into your ML pipeline")


if __name__ == "__main__":
    main()
