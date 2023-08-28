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



# random select img_ids from cluster_ids group by cluster_id limit by limit
def select_image_ids(file_name,cluster_ids,limit,url):
    # 读取label_to_id.txt文件
    cluster_dict = {cluster_id: [] for cluster_id in cluster_ids}
    with open(file_name, 'r') as f:
        for line in f:
            img_id, cluster_id = line.strip().split('\t')
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


# identify add/delete of each topic
# add pair to must_link array
# delete pair to cannot_link array
# copkmeans, pairwise constraints on inited cluster
def constraints_generate(old_clusters, new_clusters,cl=False):
    ml = []
    cl = []
    for cluster_id in set(old_clusters.keys()).union(new_clusters.keys()):
        old_patches = old_clusters.get(cluster_id, [])
        new_patches = new_clusters.get(cluster_id, [])
        half_old_patches = random.sample(old_patches, len(old_patches)//2)

        added_patches = [patch for patch in new_patches if patch not in old_patches]
        for new_patch in added_patches:
            for old_patch in half_old_patches:
                ml.append((new_patch, old_patch))
        if cl:
            removed_patches = [patch for patch in old_patches if patch not in new_patches]
            for removed_patch in removed_patches:
                half_old_patches = random.sample([patch for patch in old_patches if patch != removed_patch], len(old_patches)//2)
                for old_patch in half_old_patches:
                    cl.append((removed_patch, old_patch))

    print("old   ",old_clusters)
    print("new    ",new_clusters)
    print("must link   ",ml)
    print("cannot link ",cl)
    
    return ml, cl


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