
import math as mt
import pandas
from sklearn.model_selection import train_test_split
import Recommenders
import numpy as np
from scipy.sparse import csc_matrix
from sparsesvd import sparsesvd
triplets_file = 'song.csv'
songs_metadata_file = 'song_data.csv'
song_df_1 = pandas.read_table(triplets_file,header=None,sep='\t')

song_df_1.columns = ['user_id', 'song_id', 'listen_count']

song_df_2 =  pandas.read_csv(songs_metadata_file)
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
song_grouped = song_df.groupby(['song_id']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song_id'], ascending = [0,1])


train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)




pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song_id')


user_id = 'b80344d063b5ccb3212f76538f3d9e43d87dca9e'
recom=pm.recommend(user_id)
class popularity_recommender_py():    
    def __init__(self):        
    	self.train_data = None        
    	self.user_id = None        
    	self.item_id = None        
    	self.popularity_recommendations = None            

      
    def create(self, train_data, user_id, item_id): 
        self.train_data = train_data
        self.user_id = user_id        
        self.item_id = item_id         
        
        
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()        
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)            

        
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])            

        
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

       
        self.popularity_recommendations = train_data_sort.head(10)     

        #Use the popularity based recommender system model to    
            

    def recommend(self, user_id):            

        user_recommendations = self.popularity_recommendations                 

        
        user_recommendations['user_id'] = user_id            

        #Bring user_id column to the front        
        cols = user_recommendations.columns.tolist()        
        cols = cols[-1:] + cols[:-1]        
        user_recommendations = user_recommendations[cols]
        return user_recommendations

print(recom)
#constants defining the dimensions of our User Rating Matrix (URM) 
MAX_PID = 4 
MAX_UID = 5  

#Compute SVD of the user ratings matrix 

def computeSVD(urm, K):     
    U, s, Vt = sparsesvd(urm, K)      
    dim = (len(s), len(s))     
    S = np.zeros(dim, dtype=np.float32)     
    for i in range(0, len(s)):         
        S[i,i] = mt.sqrt(s[i])      
        U = csc_matrix(np.transpose(U), dtype=np.float32)     
        S = csc_matrix(S, dtype=np.float32)     
        Vt = csc_matrix(Vt, dtype=np.float32)          
        return U, S, Vt

def computeEstimatedRatings(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt

    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        #we convert the vector to dense format in order to get the     #indices
        #of the movies with the best estimated ratings 
        estimatedRatings[userTest, :] = prod.todense()
        recom = (-estimatedRatings[userTest, :]).argsort()[:250]
    return recom
#Used in SVD calculation (number of latent factors)
K=2

#Initialize a sample user rating matrix
urm = np.array([[3, 1, 1, 3],[4, 3, 2, 3],[3, 2, 1, 5], [1, 6, 1, 2], [5, 0,0 , 0]])
urm = csc_matrix(urm, dtype=np.float32)

#Compute SVD of the input user ratings matrix
U, S, Vt = computeSVD(urm, K)

#Test user set as user_id 4 with ratings [0, 0, 5, 0]
uTest = [2]


#Get estimated rating for test user
print("Predictied No.of times user would listen to first 4 songs:")
uTest_recommended_items = computeEstimatedRatings(urm, U, S, Vt, uTest, K, True)
print(uTest_recommended_items)
