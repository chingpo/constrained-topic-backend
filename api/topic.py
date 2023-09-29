from flask import Flask, jsonify, request
import json
from common.mysql_operate import db
import datetime
import sys
from api.helper import create_token, validate_token
from flask_cors import CORS
from config.setting import SECRET_KEY,IMG_HOME_URL,CL_FLAG,TOPIC_NUM
from api.helper import constraints_generate,update_nn,select_image_ids
from flask_jwt_extended import create_access_token, jwt_required, set_access_cookies, get_jwt_identity,JWTManager
from copkmeans.cop_kmeans import cop_kmeans
import logging
logger = logging.getLogger('my_logger')
import os
import shutil
from joblib import load

app = Flask(__name__)
CORS(app,supports_credentials=True)
app.config["JWT_SECRET_KEY"] = SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(seconds=3600*2)  # 3600 seconds 1 hour
jwt = JWTManager(app)
import numpy as np
# all vectors stacked
# vectors=np.load('copkmeans/patch_features_init.npy',allow_pickle=True) #rgb
vectors=np.load('copkmeans/patch_features_init_L.npy',allow_pickle=True) #gray 64d
vectors_2d=np.load('copkmeans/features_tsne.npy',allow_pickle=True)
# vectors_2d=np.load('copkmeans/features_pca.npy',allow_pickle=True)
#db connect test
@app.route("/users", methods=["GET"])
def get_users():
    # logger.debug("log level: DEBUG")
    # logger.info("log level:INFO")
    # logger.warning("log level:WARNING")
    # logger.error("log level:ERROR")
    # logger.critical("log level:CRITICAL")
    # sql = "SELECT * FROM user where finish_time is not null"
    sql = "SELECT user_id,comment,start_rate,end_rate,compare_rate FROM user where user_id >=55 and user_id<=60"
    data = db.select_db(sql)
    # comment = data[0]['comment']
    # print(f"Comment: {comment}")
    # comment_json = json.loads(comment)
    # print(f"Comment: {comment_json}")
    print("get all user info == >> {}".format(data))
    return jsonify({"code": 0, "data": data, "msg": "success"})




# user_id,round 放到前端
# read ids to label from np and return 
@app.route("/cluster", methods=["POST"])
@jwt_required()  
def get_clusters(): 
    user_id = get_jwt_identity().split("_")[1] 
    json_data = request.get_json()  
    round =json_data.get("round")
    if round<1:
        return jsonify({"code": 30007,  "msg": "round cannot be less than 1."})  
    file_name=os.path.join('textfile',f'nn_label_{user_id}_{round}.npy')
    info = np.load(file_name,allow_pickle=True).tolist()
    # 将结果转换为JSON
    return jsonify({"code": 0, "data": info, "msg": "success"})

# header：user_id, cluster_ids, limit, round
# random select num of limit patches 
# read ids to label from np and return 
@app.route("/patches", methods=["POST"])
@jwt_required()  
def get_patches():
    """ random get number of limit patches of each cluster """ 
    user_id = get_jwt_identity().split("_")[1]
    json_data = request.get_json()
    round =json_data.get("round")
    cluster_ids = json_data.get("cluster_ids")
    limit = json_data.get("limit")
    if(round>0):
        file_name = f'textfile/id_to_label_{user_id}.npy'
    else:
        file_name = f'textfile/id_to_label.npy'
    patches=select_image_ids(file_name,cluster_ids,limit,IMG_HOME_URL)
    return jsonify({"code": 0, "data": {'items':patches}, "msg": "success"})

# get user feedback to compute constraints and copkmeans
@app.route("/cluster-update", methods=["POST"])
@jwt_required()  
def cluster_update():
    """copkmeans""" 
    # 每个user和每次round都会修改读取txt和json
    center_k=TOPIC_NUM
    json_data = request.get_json()
    user_id = get_jwt_identity().split("_")[1]
    old_items=json_data.get("old_items") 
    new_items=json_data.get("new_items") 
    round=json_data.get("round")
    logger.info("This round %s user %s update cluster list %s", round, user_id, list(old_items.keys()))
    # update后更新nn和old items里的图片的label/cluster id
    user_label_file_path = os.path.join('textfile', f'id_to_label_{user_id}.npy')
    if round>1:
        nn_id_to_label=np.load(f'textfile/nn_label_{user_id}_{round-1}.npy',allow_pickle=True)
        with open(user_label_file_path, 'rb') as f:
            ids_in_txt = np.load(f, allow_pickle=True)
    elif round<=1:
        nn_id_to_label=np.load('textfile/nn_label_0_tsne.npy',allow_pickle=True) # nearest neighbor json
        ids_in_txt = np.load('textfile/id_to_label.npy',allow_pickle=True)
    # 获取要处理的图片的feature stack起来 
    # display
    indices = [np.where(ids_in_txt[:, 0] == item['img_name'].replace('.jpg', ''))[0][0] for item in nn_id_to_label]
    nn_selected_vectors = vectors[indices]
    # dnd
    dnd_names=sum(old_items.values(),[])
    dnd_indices = [index for name in dnd_names for index, id in enumerate(ids_in_txt) if id[0] == name]
    dnd_indices = [id for id in dnd_indices if id not in indices]
    dnd_selected_vectors = [vectors[index] for index in dnd_indices]
    data_source = np.vstack((nn_selected_vectors, dnd_selected_vectors))
    cop_index=indices+dnd_indices
    ml,cl=constraints_generate(user_id,round,old_items,new_items,ids_in_txt,cop_index,cl_flag=CL_FLAG)
    print("before copkmeans ",datetime.datetime.now())
    clusters,_ = cop_kmeans(dataset=data_source, k=center_k, ml=ml,cl=cl)
    print("after copkmeans",datetime.datetime.now())
    # 保存ids_in_txt到文件
    for index, cluster in zip(indices, clusters[:len(indices)]):
        ids_in_txt[index, 1] = str(cluster) 
    for index, cluster in zip(dnd_indices, clusters[len(indices):]):
        ids_in_txt[index, 1] = str(cluster) 
    np.save(user_label_file_path, np.array(ids_in_txt, dtype=object))
    names= [row[0] for row in ids_in_txt]
    nn_data=update_nn(names,vectors_2d,center_k,ids_in_txt)
    np.save(f'textfile/nn_label_{user_id}_{round}.npy', nn_data)
    return jsonify({"code": 0, "data": [], "msg": "success"})



@app.route("/rate", methods=["POST"])
@jwt_required()  
def rate():
    user_id = get_jwt_identity().split("_")[1]
    json_data = request.get_json()
    rate_type = json_data.get("type")  
    likert = json_data.get("likert") 
    finish_time = datetime.datetime.now()
    if rate_type is None or likert is  None:
        return jsonify({"code": 70004, "data":0, "msg": "error,type or likert cannot be none!"})
    if rate_type == "start":
        sql = "Update user set start_rate='{}' where (user_id='{}' and finish_time is null)".format(likert, user_id)
    elif rate_type== "end":  
        print(json_data.get("comment"))
        comment = json.dumps(json_data.get("comment"))
        # comment=None
        sql = "Update user set end_rate='{}' , finish_time='{}',comment='{}' where (user_id='{}' and finish_time is null)".format(likert,finish_time,comment, user_id)
    elif rate_type== "compare":
        sql = "Update user set compare_rate='{}' where (user_id='{}' and finish_time is null)".format(likert, user_id)
    data = db.execute_db(sql,"UPDATE")
    print("db return == >> {}".format(data))
    if data['status']==-1:
        return jsonify({"code": 7001,  "msg": format(data['err'])})
    elif data['status']==0: 
        if rate_type=="start":
            return jsonify({ 'code': 0, 'msg': "init rated."})
        elif rate_type=="end":
            # unset_jwt_cookies()
            return jsonify({ 'code': 0, 'msg': "task finished."})
        elif rate_type=="compare":
            return jsonify({ 'code': 0, 'msg': "copkmeans rated."})



@app.route("/register", methods=['POST'])
def user_register():
    """login""" 
    age = request.json.get("age")  # age
    gender = request.json.get("gender") # 0,1 male,2 famel
    timestamp = datetime.datetime.now()
    if age is not None and gender is not None: 
        # if (gender not in [0,1,2]):
        #     print("gender  ",gender)
        #     return jsonify({"code": 2003, "msg": "gender only in (0,1,2)"})
        # else:
        sql = "INSERT INTO user(age,gender,create_time) " \
                "VALUES('{}', '{}','{}')".format(age, gender,timestamp)
        data=db.execute_db(sql,"INSERT")
        # print("MYSQL INSERT new user ==>> {}".format(sql))
        if data['status']==-1:
            return jsonify({"code": 4001,  "msg": format(data['err'])})
        elif data['status']==0:  
            access_token = create_access_token(identity='DnD_'+str(data['lastrowid']))
            response=jsonify({"code": 0, "data":{"user_id":data['lastrowid'],"access_token":access_token},"msg": "success, user created!"})
            # unsafe
            set_access_cookies(response, access_token)
            return response
    else:
        return jsonify({"code": 2001,  "msg": "Age and gender cannot be empty."})




    
    




if __name__ == "__main__":
    app.run()  # Flask
