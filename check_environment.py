import os
import sys
import subprocess
import ctypes

print("=== Environment Diagnostic Script ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")

# Check if running as admin
try:
    is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    print(f"Running as admin: {is_admin}")
except:
    print("Could not determine admin status")

# Check for Visual C++ Runtime
print("\n=== Checking for Visual C++ Runtime ===")
system32_path = r"C:\Windows\System32"
dlls_to_check = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]

for dll in dlls_to_check:
    dll_path = os.path.join(system32_path, dll)
    exists = os.path.exists(dll_path)
    print(f"{dll}: {'FOUND' if exists else 'NOT FOUND'} at {dll_path}")

# Check registry for VC++ installation
print("\n=== Checking Registry for VC++ Installation ===")
try:
    import winreg
    keys_to_check = [
        r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    ]
    
    for key_path in keys_to_check:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                version = winreg.QueryValueEx(key, "Version")[0]
                print(f"VC++ Runtime found in registry: {version}")
        except:
            pass
except Exception as e:
    print(f"Could not check registry: {e}")

# Check packages directory
print("\n=== Checking packages directory ===")
app_dir = os.path.dirname(os.path.abspath(__file__))
packages_dir = os.path.join(app_dir, 'packages')
print(f"Packages directory: {packages_dir}")
print(f"Exists: {os.path.exists(packages_dir)}")

if os.path.exists(packages_dir):
    # Add to path
    sys.path.insert(0, packages_dir)
    os.environ["PATH"] = packages_dir + os.pathsep + os.environ["PATH"]
    
    # List ijson files
    ijson_path = os.path.join(packages_dir, 'ijson')
    if os.path.exists(ijson_path):
        print(f"\nijson directory contents:")
        for root, dirs, files in os.walk(ijson_path):
            level = root.replace(ijson_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

# Try to import C backend with detailed error
print("\n=== Testing ijson C backend import ===")
try:
    import ijson.backends.yajl2_c as yajl2_c
    print("SUCCESS: yajl2_c imported successfully!")
    print(f"Module location: {yajl2_c.__file__}")
except ImportError as e:
    print(f"FAILED: {e}")
    
    # Try to get more details
    try:
        import ijson.backends
        backends_path = os.path.dirname(ijson.backends.__file__)
        yajl2_c_path = os.path.join(backends_path, 'yajl2_c.pyd')
        print(f"\nChecking for yajl2_c.pyd at: {yajl2_c_path}")
        print(f"File exists: {os.path.exists(yajl2_c_path)}")
        
        if os.path.exists(yajl2_c_path):
            # Try to load the DLL directly to get more error info
            try:
                ctypes.CDLL(yajl2_c_path)
                print("DLL loads successfully with ctypes")
            except Exception as dll_error:
                print(f"DLL load error: {dll_error}")
                
                # Check dependencies
                print("\nChecking DLL dependencies with dumpbin (if available):")
                try:
                    result = subprocess.run(['dumpbin', '/dependents', yajl2_c_path], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(result.stdout)
                    else:
                        print("dumpbin not available")
                except:
                    print("Could not run dumpbin")
    except Exception as detail_error:
        print(f"Could not get detailed error info: {detail_error}")

print("\n=== End of diagnostic ===")

# Check for Azure packages
print("\n=== Checking for Azure packages ===")
packages_to_check = ['azure', 'azure.storage', 'azure.storage.blob']
for package in packages_to_check:
    try:
        module = __import__(package)
        print(f"{package}: FOUND at {module.__file__ if hasattr(module, '__file__') else 'built-in'}")
    except ImportError as e:
        print(f"{package}: NOT FOUND - {e}")

# List all packages in packages directory
print("\n=== All packages in packages directory ===")
if os.path.exists(packages_dir):
    items = os.listdir(packages_dir)
    for item in sorted(items):
        item_path = os.path.join(packages_dir, item)
        if os.path.isdir(item_path):
            print(f"  {item}/ (directory)")
        else:
            print(f"  {item} (file)")
else:
    print("Packages directory not found!")

# Check sys.path
print("\n=== Python sys.path ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}") 