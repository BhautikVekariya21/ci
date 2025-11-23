# setup_dagshub.py
import dagshub
import os
import subprocess

# Initialize DagsHub
dagshub.init(repo_owner='BhautikVekariya21', repo_name='ci', mlflow=True)

def setup_dvc_remote():
    """Configure DVC remote for DagsHub"""
    
    print("🔧 Configuring DVC for DagsHub...")
    
    # Get DagsHub token from environment or prompt
    token = os.getenv('DAGSHUB_TOKEN')
    if not token:
        token = input("Enter your DagsHub token: ")
    
    commands = [
        ['dvc', 'remote', 'remove', 'origin'],
        ['dvc', 'remote', 'add', 'origin', 'https://dagshub.com/BhautikVekariya21/ci.dvc'],
        ['dvc', 'remote', 'modify', 'origin', 'auth', 'basic'],
        ['dvc', 'remote', 'modify', 'origin', 'user', 'BhautikVekariya21'],
        ['dvc', 'remote', 'modify', 'origin', 'password', token],
        ['dvc', 'remote', 'default', 'origin'],
    ]
    
    for cmd in commands:
        try:
            subprocess.run(cmd, check=False, capture_output=True)
        except Exception as e:
            pass
    
    print("✅ DVC remote configured for DagsHub")

def main():
    print("=" * 70)
    print("🚀 DagsHub Setup for Sentiment Analysis Project")
    print("=" * 70)
    
    # Setup DVC
    setup_dvc_remote()
    
    print("\n" + "=" * 70)
    print("✅ DagsHub setup complete!")
    print("=" * 70)
    print("\n📊 MLflow Tracking: https://dagshub.com/BhautikVekariya21/ci.mlflow")
    print("📦 DVC Storage: https://dagshub.com/BhautikVekariya21/ci.dvc")
    print("🌐 Repository: https://dagshub.com/BhautikVekariya21/ci")
    print("\nNext steps:")
    print("  1. Run: dvc repro")
    print("  2. Run: dvc push")
    print("  3. View results on DagsHub!")

if __name__ == "__main__":
    main()