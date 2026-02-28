import pickle

pkl_path = "/home/jl17265/act7/labdata/ur_data/episode_0/trajectory.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(type(data))

if isinstance(data, dict):
    print("keys:", data.keys())
    for k, v in data.items():
        try:
            print(k, type(v), getattr(v, "shape", None))
        except:
            print(k, type(v))
elif isinstance(data, list):
    print("len:", len(data))
    print("first item type:", type(data[0]))
    if isinstance(data[0], dict):
        print("first item keys:", data[0].keys())
        for k, v in data[0].items():
            print(k, type(v), getattr(v, "shape", None))