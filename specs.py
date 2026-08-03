import sys
import platform

print("="*60)
print("             SOFTWARE & PIPELINE STACK               ")
print("="*60)
print(f"Python Version:    {sys.version.split()[0]}")
print(f"Operating System:  {platform.system()} {platform.release()}")

libs = ['pandas', 'numpy', 'scikit-learn', 'category_encoders', 'scipy']
for lib in libs:
    try:
        if lib == 'scikit-learn':
            import sklearn
            print(f"scikit-learn:      {sklearn.__version__}")
        elif lib == 'category_encoders':
            import category_encoders
            print(f"category_encoders: {category_encoders.__version__}")
        else:
            mod = __import__(lib)
            print(f"{lib:<18} {mod.__version__}")
    except ImportError:
        print(f"{lib:<18} NOT INSTALLED")
print("="*60)
