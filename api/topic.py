from flask import Flask, jsonify, request
from common.mysql_operate import db
import datetime
import sys
from api.helper import create_token, validate_token
from flask_cors import CORS
from config.setting import SECRET_KEY,IMG_HOME_URL,CL_FLAG
from api.helper import constraints_generate,dimension_reduction,select_image_ids
from flask_jwt_extended import create_access_token, jwt_required, set_access_cookies, get_jwt_identity,unset_jwt_cookies, verify_jwt_in_request,JWTManager
from copkmeans.cop_kmeans import cop_kmeans

app = Flask(__name__)
CORS(app,supports_credentials=True)
app.config["JWT_SECRET_KEY"] = SECRET_KEY
app.config['JWT_EXPIRATION_DELTA'] = datetime.timedelta(seconds=7200)
jwt = JWTManager(app)
import numpy as np
vectors=np.load('api/img_vectors.npy')
with open('api/id_to_label.txt', 'r') as f:
    ids_in_txt = [line.strip().split('\t')[0] for line in f]

#db connect test
@app.route("/users", methods=["GET"])
def get_users():
    sql = "SELECT * FROM user order by create_time desc limit 10"
    data = db.select_db(sql)
    # data=0
    print("get all user info == >> {}".format(data))
    return jsonify({"code": 0, "data": data, "msg": "success"})




# user_id,round 放到前端
# cluster 2d compute
# tsne,umap
# 前端处理逻辑，不需要api？
@app.route("/cluster", methods=["POST"])
# @jwt_required()  
def get_clusters_json_url():    
    round =request.get_json("round")
    user_id = get_jwt_identity().split("_")[1]
    if(round==0):
         file_name="nn_image_tsne.json"
    else:
         file_name=user_id+"_"+round+"nn_image_tsne.json"
    return jsonify({"code": 0, "data": file_name, "msg": "success"})

# header：user_id, cluster_ids, limit, round
# random select num of limit patches 
@app.route("/patches", methods=["POST"])
# @jwt_required()  
def get_patches():
    """ random get number of limit patches of each cluster """ 
    # user_id = get_jwt_identity().split("_")[1]
    json_data = request.get_json()
    # round =json_data.get_json("round")
    cluster_ids = json_data.get("cluster_ids")
    print(cluster_ids)
    limit = json_data.get("limit")
    file_name='/Users/seihaen/Desktop/code/backend/api/id_to_label.txt'
    # dnd读txt label to id file？还是读数据库，哪个块
    patches=select_image_ids(file_name,cluster_ids,limit,IMG_HOME_URL)
    return jsonify({"code": 0, "data": {'items':patches}, "msg": "success"})

# get user feedback to compute constraints and copkmeans
@app.route("/cluster-update", methods=["POST"])
# @jwt_required()  
def cluster_update():
    """copkmeans""" 
    # return json file name
    # display里tsne降维centers，
    # 每个user和每次round都会修改读取
    # json_data = request.get_json()
    # old_items=json_data.get("old_items") 
    # new_items=json_data.get("new_items") 
    old_items=[{'cluster_id': '1', 'id': '222_1_1', 'patches': [{'img_id': '1', 'url': 'http://136.187.116.134:18080/web/images/1_1.jpg'}, {'img_id': '2', 'url': 'http://136.187.116.134:18080/web/images/1_2.jpg'}, {'img_id': '11', 'url': 'http://136.187.116.134:18080/web/images/3_1.jpg'}, {'img_id': '12', 'url': 'http://136.187.116.134:18080/web/images/3_2.jpg'}]}, {'cluster_id': '5', 'id': '222_1_2', 'patches': [{'img_id': '3', 'url': 'http://136.187.116.134:18080/web/images/1_3.jpg'}, {'img_id': '4', 'url': 'http://136.187.116.134:18080/web/images/1_4.jpg'}]}, {'cluster_id': '6', 'id': '222_1_3', 'patches': [{'img_id': '5', 'url': 'http://136.187.116.134:18080/web/images/1_5.jpg'}, {'img_id': '6', 'url': 'http://136.187.116.134:18080/web/images/1_6.jpg'}]}, {'cluster_id': '43', 'id': '222_1_4', 'patches': [{'img_id': '7', 'url': 'http://136.187.116.134:18080/web/images/2_1.jpg'}, {'img_id': '8', 'url': 'http://136.187.116.134:18080/web/images/2_2.jpg'}]}, {'cluster_id': '3', 'id': '222_1_5', 'patches': [{'img_id': '9', 'url': 'http://136.187.116.134:18080/web/images/2_3.jpg'}, {'img_id': '10', 'url': 'http://136.187.116.134:18080/web/images/2_4.jpg'}]}]
    new_items= [{'cluster_id': '1', 'id': '222_1_1', 'patches': [{'img_id': '2', 'url': 'http://136.187.116.134:18080/web/images/1_2.jpg'}, {'img_id': '11', 'url': 'http://136.187.116.134:18080/web/images/3_1.jpg'}, {'img_id': '12', 'url': 'http://136.187.116.134:18080/web/images/3_2.jpg'}]}, {'cluster_id': '5', 'id': '222_1_2', 'patches': [{'img_id': '1', 'url': 'http://136.187.116.134:18080/web/images/1_1.jpg'}, {'img_id': '5', 'url': 'http://136.187.116.134:18080/web/images/1_5.jpg'}, {'img_id': '3', 'url': 'http://136.187.116.134:18080/web/images/1_3.jpg'}, {'img_id': '4', 'url': 'http://136.187.116.134:18080/web/images/1_4.jpg'}]}, {'cluster_id': '6', 'id': '222_1_3', 'patches': [{'img_id': '6', 'url': 'http://136.187.116.134:18080/web/images/1_6.jpg'}, {'img_id': '9', 'url': 'http://136.187.116.134:18080/web/images/2_3.jpg'}]}, {'cluster_id': '43', 'id': '222_1_4', 'patches': [{'img_id': '7', 'url': 'http://136.187.116.134:18080/web/images/2_1.jpg'}, {'img_id': '8', 'url': 'http://136.187.116.134:18080/web/images/2_2.jpg'}]}, {'cluster_id': '3', 'id': '222_1_5', 'patches': [{'img_id': '10', 'url': 'http://136.187.116.134:18080/web/images/2_4.jpg'}]}]
    old_clusters = {item['cluster_id']: [int(patch['img_id']) for patch in item['patches']] for item in old_items}
    new_clusters = {item['cluster_id']: [int(patch['img_id']) for patch in item['patches']] for item in new_items}
    ml,cl=constraints_generate(old_clusters,new_clusters,cl=CL_FLAG)
    # partial_dataset=selected_topic_item(old_clusters.keys(),vectors)
    print("before copkmeans ",datetime.datetime.now())
    clusters, centers = cop_kmeans(dataset=vectors, k=30, ml=ml,cl=cl)
    print("after copkmeans",datetime.datetime.now())
    with open('id_to_label_new.txt', 'w') as f:
        for id, center in zip(ids_in_txt, clusters):
            f.write(f'{id}\t{center}\n')
    # method 改成run时得到的参数
    chart_data=dimension_reduction(vectors,centers,20,'tsne')
    print(datetime.datetime.now())
    # 第二次round不会把上一次round的constraints效果失效么
    print(clusters)
    return jsonify({"code": 0, "data": {"clusters":0,"centers":0,"chart_data":chart_data}, "msg": "success"})


@app.route("/rate", methods=["POST"])
# @jwt_required()  
def rate():
    # user_id = get_jwt_identity().split("_")[1]
    user_id = 39
    json_data = request.get_json()
    rate_type = json_data.get("type")  
    likert = json_data.get("likert") 
    finish_time = datetime.datetime.now()
    print(finish_time)
    if rate_type is None or likert is  None:
        return jsonify({"code": 70004, "data":0, "msg": "error,type or likert cannot be none!"})
    if rate_type == "start":
        sql = "Update user set start_rate='{}' where (user_id='{}' and finish_time is null)".format(likert, user_id)
    elif rate_type== "end":    
        sql = "Update user set end_rate='{}' , finish_time='{}' where (user_id='{}' and finish_time is null)".format(likert,finish_time, user_id)
    data = db.execute_db(sql,"UPDATE")
    print("db return == >> {}".format(data))
    if data['status']==-1:
        return jsonify({"code": 7001,  "msg": format(data['err'])})
    elif data['status']==0: 
        if rate_type=="start":
            return jsonify({ 'code': 0, 'msg': "kmeans rated."})
        elif rate_type=="end":
            # unset_jwt_cookies()
            return jsonify({ 'code': 0, 'msg': "task finished."})


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
            set_access_cookies(response, access_token)
            return response
    else:
        return jsonify({"code": 2001,  "msg": "Age and gender cannot be empty."})




    
    




if __name__ == "__main__":
    app.run()  # Flask
