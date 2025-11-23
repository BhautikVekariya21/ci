# fix_dagshub_auth.py
import subprocess
import getpass

def fix_dagshub_auth():
    print("="*70)
    print("🔧 Fix DagsHub Authentication")
    print("="*70)
    
    username = input("\nEnter your DagsHub username [BhautikVekariya21]: ").strip() or "BhautikVekariya21"
    
    print("\n📝 Get your token from: https://dagshub.com/user/settings/tokens")
    token = getpass.getpass("Enter your DagsHub token: ").strip()
    
    if not token:
        print("❌ Token is required!")
        return
    
    print("\n🔧 Configuring DVC...")
    
    commands = [
        ['dvc', 'remote', 'modify', 'origin', '--local', 'auth', 'basic'],
        ['dvc', 'remote', 'modify', 'origin', '--local', 'user', username],
        ['dvc', 'remote', 'modify', 'origin', '--local', 'password', token],
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {' '.join(cmd[2:4])}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            print(e.stderr)
            return
    
    print("\n" + "="*70)
    print("✅ Authentication configured!")
    print("="*70)
    print("\nNow try:")
    print("  dvc push")

if __name__ == "__main__":
    fix_dagshub_auth()