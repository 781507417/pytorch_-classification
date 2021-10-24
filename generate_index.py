import glob

dataset_list = glob.glob('MyGestureDataset\\*')
print(dataset_list)
gesture_id = 0
for item in dataset_list:
    item_list = glob.glob(item+'\\*.png')
    print(item.split('\\')[1])
    print(item_list)
    for data in item_list:
        with open('handgesture_dataset_index.txt', 'a+') as f:
            line = data + ',' + item.split('\\')[1] + ',' + str(gesture_id) + '\n'
            f.write(line)
    gesture_id += 1
