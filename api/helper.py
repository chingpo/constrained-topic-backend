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
from scipy.spatial.distance import cdist,canberra
import logging
logger = logging.getLogger('my_logger')


def distance_cal(vectors_2d,cluster_ids,ids_in_txt,nn_num):
    result = []
    for i in cluster_ids:
        # 获取当前聚类的所有点
        cluster_points = vectors_2d[ids_in_txt[:, 1] == str(i)]    
        # 计算聚类中心
        center = cluster_points.mean(axis=0)      
        # 计算所有点到聚类中心的距离
        distances = cdist(cluster_points, center.reshape(1, -1),metric='canberra')       
        # 获取距离最近的nn_num个点的索引
        if len(cluster_points) > nn_num:
            nearest_indices = distances.argsort(axis=0)[:nn_num].flatten()    
        else:
            nearest_indices = distances.argsort(axis=0)[:].flatten()
            logger.warning("limit large than cluster length")         
        # 获取这些点的图片名字
        nearest_images = ids_in_txt[ids_in_txt[:, 1] == str(i)][nearest_indices, 0]      
        # 将结果添加到列表中
        result.append(nearest_images)
    return result

def sort_by_distance(pairs, vectors_2d, num=10):
    distances = [(pair, canberra(vectors_2d[pair[0]], vectors_2d[pair[1]])) for pair in pairs]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    if num < 0:
        return [pair for pair, distance in sorted_distances[num:]]
    elif len(sorted_distances) > num:
        return [pair for pair, distance in sorted_distances[:num]]
    else:
        return [pair for pair, distance in sorted_distances]

# random select img_ids from cluster_ids group by cluster_id limit by limit
def select_image_ids(vectors_2d,ids_in_txt,cluster_ids,limit,url):
    # 从每个聚类中随机选择图像
    nearest_images_list = distance_cal(vectors_2d, cluster_ids, ids_in_txt, limit) 
    selected_items = []
    for cluster_id, nearest_images in zip(cluster_ids, nearest_images_list):
        patches = [{"img_id": img_id, "url": url + str(img_id) + ".jpg"} for img_id in nearest_images]
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
def constraints_generate(user_id,round,old_clusters, new_clusters,ids_in_txt,dnd_indices,vectors_2d,cl_flag):
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
    # Convert indices in ml and cl to indices in dnd_indices
    ml = [(dnd_indices.index(i), dnd_indices.index(j)) for i, j in ml]
    cl = [(dnd_indices.index(i), dnd_indices.index(j)) for i, j in cl]
    # 限制ml和cl的数量
    # 分别获取距离最近和最远的约束对
    ml = sort_by_distance(ml, vectors_2d)
    logger.info("This round %s user %s give must link %s", round, user_id, ml)
    cl = sort_by_distance(cl, vectors_2d, num=-10)
    logger.info("This round %s user %s give cannot link %s", round, user_id, cl)
    return ml, cl


def update_nn(names,vectors_2d,center_k,ids_in_txt,nn_num=20):
    # display计算近邻，求json，json写到文件里
    # nearest_neighbors = model.kneighbors(centers, return_distance=False)
    chart_data = []
    # 使用distance_cal函数计算最近的图像
    nearest_images_list = distance_cal(vectors_2d, list(range(center_k)), ids_in_txt, nn_num)
    for cluster_id, neighbors in enumerate(nearest_images_list):
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