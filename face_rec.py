import numpy as np
import pandas as pd
import cv2
import os
import redis

#insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

#Time
import time
from datetime import datetime

#connect to redis client
host='redis-12285.c302.asia-northeast1-1.gce.cloud.redislabs.com'
port=12285
Password='hNxYzjnYHPBEkr0EBJQhIUWZAnXwse4D'

r=redis.StrictRedis(
    host=host,
    port=port,
    password=Password
)
#Retrieve data from database
def retrieve_data(name):
    retrieve_dict=r.hgetall(name)
    retrieve_series=pd.Series(retrieve_dict)
    retrieve_series=retrieve_series.apply(lambda x:np.frombuffer(x,dtype=np.float32))
    index=retrieve_series.index
    index=list(map(lambda x: x.decode(), index))
    retrieve_series.index= index
    retrieve_df=retrieve_series.to_frame().reset_index()
    retrieve_df.columns=['name_role','facial_features']
    retrieve_df[['name','role']]=retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['name','role','facial_features']]

#configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',
                       root='Insightface_models',
                       providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640,640),det_thresh=0.5)


#Ml search Algorithm
def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=['Name','Role'],thresh=0.5):

    """"
    Cosine Similarity base search algorithm
    """

#step-1: take the data frame (Collection of data)
   # dataframe=df.copy()
#Step-2: Index face embedding from the data frame and convert into array
    x_list = dataframe[feature_column].tolist()
    x=np.asarray(x_list)

#Step-3: Cal. Cosine Similarity
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    dataframe['cosine']=similar_arr

#Step-4: Filter the data
    data_filter=dataframe.query(f'cosine >={thresh}')
    if len(data_filter)>0:
        #Step-5: Get the persons name
        data_filter.reset_index(drop=True,inplace=True)
        argmax =data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name='UNKNOWN'
        person_role='UNKNOWN'
    return person_name, person_role

###REAL TIME PREDICTION
#Saving logs for every 1 min

class RealTimePred:

    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[]) 

    def saveLogs_redis(self):
        #Step1 create a logs dataframe
        dataframe = pd.DataFrame(self.logs)
        #Step2 drop duplicates names
        dataframe.drop_duplicates('name',inplace=True)
        #Step3 Push data to redis DB
        #Encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role , ctime in zip(name_list,role_list,ctime_list):
            if name != 'UNKNOWN':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data)>0:
            r.lpush("attendance:logs", *encoded_data)
            print(encoded_data)

        self.reset_dict()


    def face_prediction(self,test_image,dataframe,feature_column,name_role=['Name','Role'],thresh=0.5):
        #step0: get current time
        current_time =str(datetime.now())
        
        #step1: take the test image and apply to insight face
        results=faceapp.get(test_image)
        test_copy =test_image.copy()

        #Step2: Use for loop and extract each embedding and pass to ml_searc_algorithm
        for res in results :
            x1,y1,x2,y2 =res['bbox'].astype(int)
            embeddings=res['embedding']
            person_name, person_role=ml_search_algorithm(dataframe,feature_column,test_vector=embeddings, name_role=name_role,thresh=thresh)
            if person_name=='UNKNOWN':
                color=(0,0,255)#bgr
            else:
                color=(0,255,0)
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            text_gen=person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)

            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
        return test_copy



### Registration Form
class RegistrationForm:

    def __init__(self):
        self.sample=0
    def reset(self):
        self.sample=0
    
    def get_embedding(self,frame):
        #Get results from insightface model
        results=faceapp.get(frame,max_num=1)
        embeddings=None
        for res in results:
            self.sample+=1
            x1,y1,x2,y2=res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

            #Put sample captured info
            text =f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            

            #facial features
            embeddings=res['embedding']

        return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):


        #validations for name, role and txt file for embeddings
        if name is not None:
            if name.strip() != '':
                key=f'{name}@{role}'

            else:
                return "Name is fale"
        else:
            return "Name is fale"
        
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        #step1 : laod "face_embedding.txt"
        x_array=np.loadtxt('face_embedding.txt',dtype=np.float32) #flat array



        #Step-2: reshape the ixray into proper array
        received_samples = int(x_array.size/512)
        x_array=x_array.reshape(received_samples,512)
        x_array=np.asarray(x_array)

        #Step-3 cal. mean embeddings
        x_mean=x_array.mean(axis=0)
        x_mean_byte=x_mean.tobytes()


        #Step-4: Save this into redis DB(saved to redis hashes)
        r.hset(name='academy:register',key=key,value=x_mean_byte)

        # Delete the embeddings file .txt
        os.remove('face_embedding.txt')
        self.reset

        return True

    
   