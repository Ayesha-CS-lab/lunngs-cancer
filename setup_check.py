"""
Quick start script to verify installation and setup.
"""
import sys
import subprocess


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required = [
        'torch',
        'torchvision',
        'efficientnet_pytorch',
        'albumentations',
        'sklearn',
        'cv2',
        'gradio'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available!")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")


def create_placeholder_dirs():
    """Create placeholder files in data directories."""
    import os
    
    print("\nCreating directory structure...")
    
    dirs = [
        'data/raw',
        'data/processed/train/no_cancer',
        'data/processed/train/cancer',
        'data/processed/val/no_cancer',
        'data/processed/val/cancer',
        'data/splits',
        'checkpoints',
        'outputs/heatmaps',
        'outputs/plots',
        'experiments'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        # Create .gitkeep
        gitkeep = os.path.join(dir_path, '.gitkeep')
        if not os.path.exists(gitkeep):
            open(gitkeep, 'w').close()
    
    print("✅ Directory structure created!")


def main():
    """Main setup verification."""
    print("="*70)
    print("LUNG CANCER AI - SETUP VERIFICATION")
    print("="*70 + "\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_placeholder_dirs()
    
    print("\n" + "="*70)
    if deps_ok:
        print("✅ SETUP COMPLETE!")
        print("\nNext steps:")
        print("1. Place your CT scan images in data/raw/")
        print("2. Organize into no_cancer/ and cancer/ folders")
        print("3. Run: python train.py --model efficientnet")
    else:
        print("❌ SETUP INCOMPLETE - Please install missing dependencies")
    print("="*70)


if __name__ == "__main__":
    main()
