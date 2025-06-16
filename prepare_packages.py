import subprocess
import sys
import os
import shutil

def prepare_packages():
    """
    Prepares a packages directory with all required dependencies for the batch job.
    This should be run on a Windows machine with the same Python version (3.10) as the target.
    """
    
    packages_dir = os.path.join(os.getcwd(), 'packages')

    # Clean up old packages directory to ensure a fresh install
    if os.path.exists(packages_dir):
        print(f"--- Removing existing packages directory: {packages_dir} ---")
        shutil.rmtree(packages_dir)

    # Create a new, empty packages directory
    print(f"--- Creating new packages directory: {packages_dir} ---")
    os.makedirs(packages_dir)
    
    # List of required packages
    required_packages = [
        'azure-storage-blob',  # This will also install azure-core and other dependencies
        'ijson',  # You already have this, but including for completeness
        'psutil',  # For memory monitoring
    ]
    
    print(f"--- Installing packages to: {packages_dir} ---")
    
    for package in required_packages:
        print(f"\nInstalling {package}...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            package, 
            '--target', packages_dir,
            '--upgrade'
        ])
    
    print("\n=== Installation complete ===")
    print(f"Packages installed to: {packages_dir}")
    print("\nContents of packages directory:")
    
    for item in sorted(os.listdir(packages_dir)):
        item_path = os.path.join(packages_dir, item)
        if os.path.isdir(item_path):
            print(f"  {item}/")
        else:
            print(f"  {item}")
    
    print("\nIMPORTANT: Make sure to include the entire 'packages' directory in your zip file.")
    print("The directory structure should be:")
    print("  your_zip.zip/")
    print("    ├── batchactivity_json2csv.py")
    print("    ├── check_environment.py")
    print("    └── packages/")
    print("        ├── azure/")
    print("        ├── azure_core-*.dist-info/")
    print("        ├── azure_storage_blob-*.dist-info/")
    print("        ├── ijson/")
    print("        └── ... (other dependencies)")

if __name__ == "__main__":
    prepare_packages() 