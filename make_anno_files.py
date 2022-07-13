import glob, os, numpy as np
task_cat2mpcat40_labels = [
    'chair',
    'table',
    'picture',
    'cabinet',
    'cushion',
    'sofa',
    'bed',
    'chest_of_drawers',
    'plant',
    'sink',
    'toilet',
    'stool',
    'towel',
    'tv_monitor',
    'shower',
    'bathtub',
    'counter',
    'fireplace',
    'gym_equipment',
    'seating',
    'clothes',
]
OBJECT_TARGET_CATEGORY = {}
OBJECT_TARGET_CATEGORY['gibson'] = ['chair', 'couch', 'potted plant', 'bed', 'toilet', 'tv']
OBJECT_TARGET_CATEGORY['mp3d'] = task_cat2mpcat40_labels

with open('/disk3/nuri/video_cluster_ckpt/gibsontiny_anno/classInd.txt', 'w') as f:
    for i, clss_name in enumerate(OBJECT_TARGET_CATEGORY['gibson']):
        f.write(f"{i+1}" + " " + clss_name.replace(" ", "_"))
        f.write('\n')
        print(i+1, clss_name)

train_list = []
objects = os.listdir('/disk3/nuri/video_cluster_ckpt/gibsontiny')
for obj in objects:
    train_list.append(glob.glob("/disk3/nuri/video_cluster_ckpt/gibsontiny" + "/" + obj + "/*"))
train_list = np.concatenate(train_list)
with open('/disk3/nuri/video_cluster_ckpt/gibsontiny_anno/trainlist.txt', 'w') as f:
    for i, train_path in enumerate(train_list):
        clss_name = train_path.split("/")[-2]
        cls_idx = OBJECT_TARGET_CATEGORY['gibson'].index(clss_name.replace("_", " ")) + 1
        f.write(train_path + " "+ str(cls_idx))
        f.write('\n')
        print(train_path, cls_idx)


os.makedirs('/disk3/nuri/video_cluster_ckpt/mp3d_anno', exist_ok=True)
with open('/disk3/nuri/video_cluster_ckpt/mp3d_anno/classInd.txt', 'w') as f:
    for i, clss_name in enumerate(OBJECT_TARGET_CATEGORY['gibson']):
        f.write(f"{i+1}" + " " + clss_name.replace(" ", "_"))
        f.write('\n')
        print(i+1, clss_name)

train_list = []
objects = os.listdir('/disk3/nuri/video_cluster_ckpt/mp3d')
for obj in objects:
    train_list.append(glob.glob("/disk3/nuri/video_cluster_ckpt/mp3d" + "/" + obj + "/*"))
train_list = np.concatenate(train_list)
with open('/disk3/nuri/video_cluster_ckpt/mp3d_anno/trainlist.txt', 'w') as f:
    for i, train_path in enumerate(train_list):
        clss_name = train_path.split("/")[-2]
        cls_idx = OBJECT_TARGET_CATEGORY['gibson'].index(clss_name.replace("_", " ")) + 1
        f.write(train_path + " "+ str(cls_idx))
        f.write('\n')
        print(train_path, cls_idx)