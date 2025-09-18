import json
with open("downloaded_map.json", "r") as file:
    data = json.load(file)
count = 0

print(len(data))

# for i,val in data.items():
#     if val == True:
#         count+=1

# print(f"Total downloaded files: {count}")
    