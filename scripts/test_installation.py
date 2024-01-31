# test if all the packages are installed correctly


packages = ['numpy', 'matplotlib', 'torch', 'dolfin', 'meshio', 'dgl', 'scipy', 'tqdm','jupyter']
flag = False
for package in packages:
    try:
        __import__(package)
    except ImportError:
        flag = True
        print(f"{package} is not installed.")
if not flag:
    print("All packages are installed correctly!")