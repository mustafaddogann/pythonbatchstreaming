import os
import sys

print("=== Safe Environment Diagnostic Script ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")

try:
    import ctypes
    # Check if running as admin
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        print(f"Running as admin: {is_admin}")
    except:
        print("Could not determine admin status")
except Exception as e:
    print(f"ctypes import failed: {e}")

# Check for Visual C++ Runtime DLLs
print("\n=== Checking for Visual C++ Runtime ===")
system32_path = r"C:\Windows\System32"
dlls_to_check = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]

for dll in dlls_to_check:
    dll_path = os.path.join(system32_path, dll)
    exists = os.path.exists(dll_path)
    print(f"{dll}: {'FOUND' if exists else 'NOT FOUND'} at {dll_path}")

# Check packages directory
print("\n=== Checking packages directory ===")
app_dir = os.path.dirname(os.path.abspath(__file__))
packages_dir = os.path.join(app_dir, 'packages')
print(f"Packages directory: {packages_dir}")
print(f"Exists: {os.path.exists(packages_dir)}")

# Check for DLLs in packages directory
if os.path.exists(packages_dir):
    print("\nDLLs in packages directory:")
    for dll in dlls_to_check:
        dll_path = os.path.join(packages_dir, dll)
        exists = os.path.exists(dll_path)
        print(f"{dll}: {'FOUND' if exists else 'NOT FOUND'} in packages")
    
    # Add to path
    sys.path.insert(0, packages_dir)
    os.environ["PATH"] = packages_dir + os.pathsep + os.environ["PATH"]

# Try to import ijson
print("\n=== Testing ijson import ===")
try:
    import ijson
    print(f"ijson imported successfully from: {ijson.__file__}")
    
    # Try C backend
    try:
        import ijson.backends.yajl2_c as yajl2_c
        print("SUCCESS: yajl2_c backend imported!")
    except Exception as e:
        print(f"yajl2_c import failed: {e}")
        
        # Check for the pyd file
        try:
            ijson_backends_path = os.path.join(packages_dir, 'ijson', 'backends')
            print(f"\nChecking backends directory: {ijson_backends_path}")
            if os.path.exists(ijson_backends_path):
                files = os.listdir(ijson_backends_path)
                pyd_files = [f for f in files if f.endswith('.pyd')]
                print(f"PYD files found: {pyd_files}")
        except Exception as list_error:
            print(f"Could not list backends directory: {list_error}")
            
except Exception as e:
    print(f"ijson import failed: {e}")

print("\n=== Script completed successfully ===") 