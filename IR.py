import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer
temmer = SnowballStemmer("english")
from math import*
import heapq

# read csv files with panda
df_books = pd.read_csv('bx_books.csv')
df_ratings = pd.read_csv('bx_books_ratings.csv')
df_users = pd.read_csv('bx_users.csv')


# executing stopwords
df_books['book_title'] = df_books['book_title'].str.lower().str.split()
stop = stopwords.words('english')
df_books['book_title'].apply(lambda x: [item for item in x if item not in stop])

# executing stemming
df_books['book_title'] = df_books['book_title'].apply(lambda x: [stemming.stem(y) for y in x])
cols_to_drop = ['book_title']
df_books.drop(cols_to_drop, inplace=True)

# combine csv files to more usefull form
df_book_profile = df_books[['book_title','book_author','year_of_publication']]
df_userbook = pd.merge(df_users, df_books, on='USEID')[['user_id','book_title','isbn','book_author','year_of_publication']]

# define jaccard similarity
def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

# define dice coefficient
def dice_coefficient(a,b):
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0
    
    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]
    
    a_bigram_list.sort()
    b_bigram_list.sort()
    
    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1
    
    score = float(2*matches)/float(lena + lenb)
    return score

# make our user profile from his/her best ratings
def user_profile(user_id):
    best = df_ratings.nlargest(3,'user_id')
    return df_book_profile[best]

# calculate both similarities 1:jaccard 2:dice coefficient
def similarity1(x,y):
    result = 0
    if jaccard_similarity(x,y):
        result += 0.2
    if df_books['author','x'] == df_books['author','y']:
        result += 0.4
    if 1-(abs(x-y)/2005)<3:
        result += 0.4
    return result , df_books['y']

def similarity2(x,y):
    result = 0
    if dice_coefficient(x,y):
        result += 0.5
    if df_books['author','x'] == df_books['author','y']:
        result += 0.3
    if 1-(abs(x-y)/2005)<3:
        result += 0.2
    return result , df_books['y']

# define recommendations for the user using the first similarity
def recommendations1(user_id):
    myList1 = []
    a = df_userbook['user_id','book_title']
    for i in range (df_books['isbn']):
        b = df_userbook['isbn','book_title']
        if a != b:
            myList1.append(similarity1(a,b))
    return heapq.nlargest(10, myList1)

# define recommendations using the second similarity
def recommendations2(user_id):
    myList2 = []
    a = df_userbook['user_id','book_title']
    for i in range (df_books['isbn']):
        b = df_userbook['isbn','book_title']
        if a != b:
            myList1.append(similarity2(a,b))
    return heapq.nlargest(10, myList2)

# picking 5 random users and give them recommendations, save the results in a list and find the average of the overlap
RandomUser = df_users['user_id'].sample(5)
Exitfile[i] = []
Overlaplist1[i]= []
for i in range (RandomUser):
    ty1 = recommendations1(RandomUser)
    ty2 = recommendations2(RandomUser)
    Exitfile[i].append(df_users['user_id'], ty1, ty2)
    print (Exitfile[i])
    get_sim_score(ty1, ty2)
    oc = OverlapCoefficient()
    Overlaplist1[i].append(c.get_sim_score([], []))

def Average(lst): 
    return sum(lst) / len(lst)
for i in range (RandomUser):
    print (Average(Overlaplist1[i]))
    
# make the golden standard for each user and each recommendation    
goldenstandard1[i] = []
goldenstandard2[i] = []
aplist1[i] = []
aplist2[i] = []

numbercount1 = 0
numbercount2 = 0
for i in range (RandomUser):
    for j in range (Exitfile[j]):
        if df_userbook['book_title'] == Exitfile[j, ty1]:
            numbercount += 1
            aplist1[i].append(numbercount1, Exitfile[j, ty1])
        if df_userbook['book_title'] == Exitfile[j, ty2]: 
            numbercount +=1
            aplist2[i].append(numbercount2, Exitfile[j, ty2])
            

for i in range (RandomUser):
    goldenstandard1[i] = heapq.nlargest(10, aplist1[i], Exitfile[i, ty1], numbercount1)
    goldenstandard2[i] = heapq.nlargest(10, aplist2[i], Exitfile[i, ty2], numbercount2)
    

# Overlap for golden standard
Overlaplist2[i]= []
for i in range (RandomUser):
    get_sim_score(goldenstandard1, goldenstandard2)
    oc = OverlapCoefficient()
    Overlaplist2[i].append(c.get_sim_score([], []))    
for i in range (RandomUser):
    print (Average(Overlaplist2[i]))
