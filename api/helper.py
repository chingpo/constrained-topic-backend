import time
import jwt
from config.setting import SECRET_KEY
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import json
import os
import uuid
from scipy.spatial.distance import cdist
import logging
logger = logging.getLogger('my_logger')



# random select img_ids from cluster_ids group by cluster_id limit by limit
def select_image_ids(file_name,cluster_ids,limit,url):
    # 读取label_to_id.txt文件
    cluster_dict = {cluster_id: [] for cluster_id in cluster_ids}
    ids_in_txt = np.load(file_name, allow_pickle=True).tolist()
    for line in ids_in_txt:
        img_id, cluster_id = line
        cluster_id = int(cluster_id)  # convert cluster_id to int
        if cluster_id in cluster_dict:
            cluster_dict[cluster_id].append(img_id)
    # 从每个聚类中随机选择图像
    selected_imgs = {}
    selected_items = []
    for cluster_id, img_ids in cluster_dict.items():
        selected_imgs={}
        selected_imgs[cluster_id] = random.sample(img_ids, limit)
        cluster_id, img_ids = list(selected_imgs.items())[0]
        patches = [{"img_id": img_id, "url": url + str(img_id) + ".jpg"} for img_id in img_ids]
        selected_items.append({
            "id": str(uuid.uuid4())[:8],  # generate a unique 8-char string
            "cluster_id": str(cluster_id),
            "patches": patches
        })
    return selected_items


# identify add/delete of each cluster
# add pair to must_link array
# delete pair to cannot_link array
# index in dnd_indices
def constraints_generate(user_id,round,old_clusters, new_clusters,ids_in_txt,dnd_indices,cl_flag):
    ml = []
    cl = []
    id_to_index = {id[0]: index for index, id in enumerate(ids_in_txt)}
    for cluster_id in set(old_clusters.keys()):
        old_patches = old_clusters.get(cluster_id, [])
        new_patches = new_clusters.get(cluster_id, [])
        half_remain_patches = [patch for patch in new_patches if patch in old_patches]  # after dnd still in the same cluster
        half_remain_patches = random.sample(half_remain_patches, len(half_remain_patches)//8) # 0.5 0.25

        added_patches = [patch for patch in new_patches if patch not in old_patches]
        for added_patch in added_patches:
            add_patch_index = id_to_index[added_patch]
            for flag_patch in half_remain_patches:
                flag_patch_index = id_to_index[flag_patch]
                ml.append((add_patch_index, flag_patch_index))
        if cl_flag:
            removed_patches = [patch for patch in old_patches if patch not in new_patches]
            for removed_patch in removed_patches:
                removed_patch_index = id_to_index[removed_patch]
                for flag_patch in half_remain_patches:
                    flag_patch_index = id_to_index[flag_patch]
                    cl.append((removed_patch_index, flag_patch_index))
    logger.info("This round %s user %s give must link %s", round, user_id, ml)
    logger.info("This round %s user %s give cannot link %s", round, user_id, cl)
    # Convert indices in ml and cl to indices in dnd_indices
    ml = [(dnd_indices.index(i), dnd_indices.index(j)) for i, j in ml]
    cl = [(dnd_indices.index(i), dnd_indices.index(j)) for i, j in cl]
    return ml, cl


def update_nn(names,vectors_2d,center_k,ids_in_txt,nn_num=20):
    # display计算近邻，求json，json写到文件里
    # nearest_neighbors = model.kneighbors(centers, return_distance=False)
    chart_data = []
    result = []
    for i in range(center_k):
        # 获取当前聚类的所有点
        cluster_points = vectors_2d[ids_in_txt[:, 1] == str(i)]     
        # 计算聚类中心
        center = cluster_points.mean(axis=0)       
        # 计算所有点到聚类中心的距离
        distances = cdist(cluster_points, center.reshape(1, -1))       
        # 获取距离最近的20个点的索引
        nearest_indices = distances.argsort(axis=0)[:nn_num].flatten()        
        # 获取这些点的图片名字
        nearest_images = ids_in_txt[ids_in_txt[:, 1] == str(i)][nearest_indices, 0]      
        # 将结果添加到列表中
        result.append(nearest_images)

    for cluster_id, neighbors in enumerate(result):
        for neighbor in neighbors:
            index = names.index(neighbor)
            chart_data.append({
                'cluster_id': cluster_id,
                'img_name': neighbor + ".jpg",
                'x': float(vectors_2d[index][0]),
                'y': float(vectors_2d[index][1])
            }) 
    return chart_data

#  vectors=np.load('./img_vectors.npy')
def dimension_reduction(vectors,centers,n_neighbors,method='tsne'):
    if(method=='tsne'):
        model = TSNE()
    elif(method=='pca'):
        model = PCA(n_components=2)
    # display计算近邻，求json，json写到文件里
    all = np.concatenate((vectors, centers))
    all_low_d = model.fit_transform(all) 
    center_low_d = all_low_d[-30:]
    vector_low_d = all_low_d[:-30] 
    nn = NearestNeighbors(n_neighbors)
    # 训练模型
    nn.fit(vectors)
    nearest_neighbors = nn.kneighbors(centers, return_distance=False)
    chart_data = []
    image_dir = "../data/image/"  
    image_names = sorted(os.listdir(image_dir), key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))
    for c, i in enumerate(vector_low_d):
        chart_data.append({
        'img_name': image_names[c],
        'cluster_id': int(centers[c]),  # 将numpy的int32转换为原生的int
        'x': float(i[0]),
        'y': float(i[1])
    })
    with open('image_pca_projections.json', 'w') as out:
        json.dump(chart_data, out)   
    return chart_data
 # 存到数据库里，写成字符串
  # 每个user和round
# 前端直接读，然后props带到跳转的页面
 



# token=create_token( data['lastrowid'])
# res.set_cookie("token", token, max_age=7200, httponly=True) # 2hours expired
def create_token(id):
    exp = int(time.time() + 7200)
    key=SECRET_KEY
    payload = {
    "user_id": id,
    "exp": exp
    }
    token = jwt.encode(payload, key, algorithm='HS256')
    return token

# token = request.cookies.get('token', None)
#     if not token:
#         return jsonify({ 'code': 3001, 'msg': "access token empty"})
#     payload, msg = validate_token(token)
#     if msg:
#         return jsonify({ 'code': 500, 'msg': msg})
# user_id = payload['user_id']
def validate_token(token):
    key=SECRET_KEY
    payload = None
    msg = None
    try:
        print(token)
        payload = jwt.decode(token, key, algorithms=['HS256'])
        print(payload)
    except jwt.exceptions.ExpiredSignatureError:
        msg = 'token expired'
    except jwt.DecodeError:
        msg = 'token illegal'
    except jwt.InvalidTokenError:
        msg = 'invalid token'
    return (payload, msg)