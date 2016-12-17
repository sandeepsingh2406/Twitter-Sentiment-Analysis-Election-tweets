import csv,re, nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from random import shuffle
from sklearn.naive_bayes import BernoulliNB


stopwords_list=[]
alltweets_list=[]
list_emoticons=[]
dict_emoticons={}

def get_stopwords(stopwords_file):
#Removing stopwords from data using stopwords in a seperate file
	global stopwords_list	
	stopwords_list = [line.rstrip('\n').lower() for line in open(stopwords_file)]
	stopwords_list.append("url")

def emoticons():
#Replacing common smileys with the representative words
#did not improve accuracy, so removed eventually by removing call to this function
		emoticons = \
		[	('EMOTSMILEY',	[':-)', ':)', '(:', '(-:', ] )	,\
			('EMOTLAUGH',		[':-D', ':D', 'X-D', 'XD', 'xD', ] )	,\
			('EMOTLOVE',		['<3', ':\*', ] )	,\
			('EMOTWINK',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ] )	,\
			('EMOTFROWN',		[':-(', ':(', '(:', '(-:', ] )	,\
			('EMOTCRY',		[':,(', ':\'(', ':"(', ':(('] )	,\
		]
		
		for word,list_emot in emoticons:
			for emot in list_emot:
				list_emoticons.append(emot)
				dict_emoticons[emot]=word.lower()
		



def clean(filename):
	# print(stopwords_list)
	stemmer=PorterStemmer();
	i=0

	global alltweets_list
	alltweets_list=[]
	with open(filename, newline='', encoding='latin1') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for line in reader:

			i=i+1
			if i<3:
				continue
#Getting rid of first 2 rows
			line[4]=line[4].strip()
			if (line[4] == "1" or line[4] == "-1" or line[4] == "0"):
				# print(line[3],line[4])
				tweet_text=line[3]
				
				
				##Removing proper nouns- caused steep decline in accuracy, so removed eventually
				# print(tweet_text)
				# for word,pos in nltk.pos_tag(nltk.word_tokenize(tweet_text)):
					# if pos == 'NNP':
						# tweet_text = tweet_text.replace(word," ")
				# print(tweet_text)
				# print(nltk.pos_tag(nltk.word_tokenize(tweet_text)))
				
				
				tweet_text=tweet_text.lower().strip()
				tweet_text=re.compile("<.*?>").sub(" ", tweet_text)
				#take out html tags
				#above line takes out some(10) phrases which could be relevant, can fix later if reqd. can see those phrases by below method
				
				tweet_text=re.compile("(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?").sub(" url ", tweet_text)#Removing URLs
				tweet_text=re.compile("@(\w)*").sub(" atuser ", tweet_text) #removing @username
				# print(tweet_text)

				tweet_text=re.compile("#(\w)+").sub(" hashtag ", tweet_text) #removing hashtag
				# hash_regex = re.compile(r"#(\w+)")
				# tweet_text=re.sub(r'#(\w+)',r' HASH_\1', tweet_text) #removing hashtag
				# print(tweet_text)
				
				# print(tweet_text)
				

                                ##getting emoticons, did not improve accuracy, so removed eventually
#				for key,value in dict_emoticons.items():
#					tweet_text=tweet_text.replace(key," "+value+" ")
				
				tweet_text=tweet_text.replace("'","").replace("."," ").replace(","," ")# also converting words like "can't" to "cant"
				tweet_text=re.compile("[^\w']|_").sub(" ", tweet_text) 
				words=tweet_text.split()
				mod_tweet_text=""
				for word in words:
					if (word.isalpha()):
						if (word not in stopwords_list) and (stemmer.stem(word) not in stopwords_list):
							# word=stemmer.stem(word)
							word=re.sub(r'(.)\1{2,}', r'\1', word)#if some alphabet repeated 3 or more times, replace by single occurence
							word=stemmer.stem(word)
							mod_tweet_text=mod_tweet_text+" "+word+" "#replacing word with stem
				#print(tweet_text)
				#tweet_text=re.compile("[^\w']|_").sub(" ", mod_tweet_text) # converting punctuations and special characters to white space
				
				# keeping only pos,neg,neutral tweets
				mod_tweet_text=re.sub(' +',' ',mod_tweet_text).strip()
				if line[4]=="1":
					label="Positive"
				elif line[4]=="-1":
					label="Negative"
				else:
					label="Neutral"

				alltweets_list.append((mod_tweet_text,label))


	#Below commented code is for sampling(for Romey data), tried over/under sampling, but either didn't help much, so removed
	#print(alltweets_list[0])
	#if(filename=="Romney.csv"):
	#    positive=[]
	#    negative=[]
	#   neutral=[]
	#    size=0     
	#    print("Sampling data for Romney") 
	#    for tweet,label in alltweets_list:
	#        if(label=="Positive"): positive.append((tweet,label))
	#        elif(label=="Negative"): negative.append((tweet,label))
	#        elif(label=="Neutral"): neutral.append((tweet,label))  	    	        
	#    print("p",len(positive),"n",len(negative),"nl",len(neutral)) 
	#    size=min(len(positive),len(negative),len(neutral))
	#    alltweets_list=[]

	#    if(len(positive)==size):
	#        for tweet,label in positive:
	#            alltweets_list.append((tweet,label))
	#    else: 
	#         df=pd.DataFrame(positive)  
	#         randomsample=df.sample(n=size,replace=False).values.tolist()
	           
	#         for tweet,label in randomsample:
	#            alltweets_list.append((tweet,label))
      
	#    if(len(negative)==size):
	#        for tweet,label in negative:
	#            alltweets_list.append((tweet,label))
	#    else: 
	#         df=pd.DataFrame(negative)  
	#         randomsample=df.sample(n=size,replace=False).values.tolist()
	           
	#         for tweet,label in randomsample:
	#            alltweets_list.append((tweet,label))

	#   if(len(neutral)==size):
	#        for tweet,label in neutral:
	#            alltweets_list.append((tweet,label))
	#    else: 
	#         df=pd.DataFrame(neutral)  
	#         randomsample=df.sample(n=size,replace=False).values.tolist()
	           
	#         for tweet,label in randomsample:
	#           alltweets_list.append((tweet,label))


            				


def cross_validation(all_tweets,person_name):
    avg_accuracy=0
    print("Cross Validation starting for "+person_name)
    nFold = 10    
    pos_precision=0
    neg_precision=0
    pos_recall=0
    neg_recall=0
    subset_size = int(len(all_tweets)/nFold)
    pos_fscore=0
    neg_fscore=0
    print(subset_size)
    for j in range(nFold):
        correct=0
        test_tweets = []
        train_tweets = []
        print("Iteration:",j)        
        tp=0
        tn=0
        actual_positive=0
        predicted_positive=0
        actual_negative=0
        predicted_negative=0
        
        # selecting training and validation tweets
        test_tweets = all_tweets[j*subset_size:][:subset_size]
        train_tweets = all_tweets[:j*subset_size] + all_tweets[(j+1)*subset_size:]  	
        
        #creating tfidf vector
        data = []
        target_names = []
        for i in range(0, len(train_tweets)):
            data.append(train_tweets[i][0])
            target_names.append(train_tweets[i][1]) 
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(data)    
        tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        
        ###################################################################### 
        #using classifier svm
        clf_svm = svm.SVC(C=1.1,gamma=0.1,kernel='linear')
        #clf_svm.fit(X_train_tf, target_names)
        ##grid search param tuning
        # C_range = 10.0 ** np.arange(-4, 4)
        # gamma_range = 10.0 ** np.arange(-4, 4)
        # param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())
        # print(param_grid)
        # clf = svm.SVC()
        # grid = GridSearchCV(clf, param_grid)
        # grid.fit(X_train_tf, target_names)
        
        #using BernoulliNB
        clf_bnb = BernoulliNB()        
        ######################################################################
        #using classifier RandomForest
        clf_rf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                      min_samples_split=10, random_state=0)
        #clf_rf.fit(X_train_tf, target_names)
		
        #####################################################################
        #using stochastic gradient descent
        clf_sgd = SGDClassifier(loss="log", penalty="l2")
        #clf_sgd.fit(X_train_tf, target_names)
        
        #####################################################################
        #using Logistic Regression
        clf_lr = LogisticRegression()
        #clf_lr.fit(X_train_tf, target_names)
        #####################################################################
        #using knn - Least accuracy obtained so removed from the voting classifier
        #clf_knn = KNeighborsClassifier()
        #clf_knn.fit(X_train_tf, target_names) 
        #####################################################################
        #using ensemble classifier- votingClassifier with equal weights to all the 
        #classifiers
        eclf = VotingClassifier(estimators = [('svm', clf_svm), ('gnb', clf_bnb), ('sgd', clf_sgd), ('lr', clf_lr), ("rf", clf_rf)])
        eclf.fit(X_train_tf, target_names) 
        correct = 0
        
        #testing accuracy on the validation set
        X_test_counts = count_vect.transform(t[0] for t in test_tweets)
        X_test_data = tf_transformer.transform(X_test_counts)
        classified = eclf.predict(X_test_data)
        for i in range(0, len(test_tweets)):
       
            if classified[i] == test_tweets[i][1]:
                correct = correct + 1
                if classified[i] == "Positive":
                    tp=tp+1
                elif classified[i] == "Negative":
                    tn=tn+1
    
            if test_tweets[i][1] == "Positive":
                actual_positive=actual_positive+1
            elif test_tweets[i][1] == "Negative":
                actual_negative=actual_negative+1
    
            if classified[i] == "Positive":
                predicted_positive=predicted_positive+1
            elif classified[i] == "Negative":
                predicted_negative=predicted_negative+1
    
    	
        # printing all numbers for each k-fold iteration    	
        print("Accuracy : ")
        print(correct/len(test_tweets)) 
        avg_accuracy=avg_accuracy+(correct/len(test_tweets))
    
        print("Positive Precision : ")
        pos_precision=pos_precision+(tp/predicted_positive)
        current_pos_precision=(tp/predicted_positive)
        print(tp/predicted_positive)

        print("Negative Precision : ")
        neg_precision=neg_precision+(tn/predicted_negative)
        current_neg_precision=(tn/predicted_negative)
        print((tn/predicted_negative))    
    
        print("Positive Recall : ")
        pos_recall=pos_recall+(tp/actual_positive)
        current_pos_recall=(tp/actual_positive)
        print(tp/actual_positive)
    
        print("Negative Recall : ")
        neg_recall=neg_recall+(tn/actual_negative)
        current_neg_recall=(tn/actual_negative)
        print((tn/actual_negative))  

        print("Positive class F-score : ")
        pos_fscore=pos_fscore+( (2*current_pos_precision*current_pos_recall)/ ( (current_pos_precision)+(current_pos_recall) ) )
        print(( (2*current_pos_precision*current_pos_recall)/ ( (current_pos_precision)+(current_pos_recall) ) ))
		
        print("Negative class F-score : ")
        neg_fscore=neg_fscore+( (2*current_neg_precision*current_neg_recall)/ ( (current_neg_precision)+(current_neg_recall) ) )
        print(( (2*current_neg_precision*current_neg_recall)/ ( (current_neg_precision)+(current_neg_recall) ) ))
    	
    # printing all the average numbers	
    print("Final Accuracy",avg_accuracy/nFold)
    print("Average Positive Precision",pos_precision/nFold)
    print("Average Negative Precision",neg_precision/nFold)
    print("Average Positive Recall",pos_recall/nFold)
    print("Average Negative Recall",neg_recall/nFold)    
    print("Average Positive class F-Score",( pos_fscore/nFold))
    print("Average Negative class F-Score",( neg_fscore/nFold))
				

def test_data_classification(train_tweets,test_tweets,person_name):	
    avg_accuracy=0
    print("Test data Classification starting for "+person_name)

    pos_precision=0
    neg_precision=0
    pos_recall=0
    neg_recall=0
    correct=0
    tp=0
    tn=0
    actual_positive=0
    predicted_positive=0
    actual_negative=0
    predicted_negative=0
	
    #creating tfidf vector
    data = []
    target_names = []
    for i in range(0, len(train_tweets)):
        data.append(train_tweets[i][0])
        target_names.append(train_tweets[i][1]) 
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)    
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
	
    #using BernoulliNB
    clf_bnb = BernoulliNB()  
    ###################################################################### 
    #using classifier svm
    clf_svm = svm.SVC(C=1.1,gamma=0.1,kernel='linear')
    #clf_svm.fit(X_train_tf, target_names)
    ##grid search param tuning
    # C_range = 10.0 ** np.arange(-4, 4)
    # gamma_range = 10.0 ** np.arange(-4, 4)
    # param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())
    # print(param_grid)
    # clf = svm.SVC()
    # grid = GridSearchCV(clf, param_grid)
    # grid.fit(X_train_tf, target_names)
    
    ######################################################################
    #using classifier RandomForest
    clf_rf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                  min_samples_split=10, random_state=0)
        #clf_rf.fit(X_train_tf, target_names)
		
    #####################################################################
    #using stochastic gradient descent
    clf_sgd = SGDClassifier(loss="log", penalty="l2")
    #clf_sgd.fit(X_train_tf, target_names)
    
    #####################################################################
    #using Logistic Regression
    clf_lr = LogisticRegression()
    #clf_lr.fit(X_train_tf, target_names)
    #####################################################################
    #using knn - Least accuracy obtained so removed from the voting classifier
    #clf_knn = KNeighborsClassifier()
    #clf_knn.fit(X_train_tf, target_names) 
    #####################################################################
    #using ensemble classifier- votingClassifier with equal weights to all the 
    #classifiers
    eclf = VotingClassifier(estimators = [('svm', clf_svm), ('bnb', clf_bnb), ('sgd', clf_sgd), ('lr', clf_lr), ("rf", clf_rf)])
    #Romney gives better F-Score with the below voting classifier
    #eclf = VotingClassifier(estimators = [('svm', clf_svm), ('bnb', clf_bnb), ('sgd', clf_sgd)])
    eclf.fit(X_train_tf, target_names)  
    correct = 0
        
    #testing accuracy on the validation set
    X_test_counts = count_vect.transform(t[0] for t in test_tweets)
    X_test_data = tf_transformer.transform(X_test_counts)
    classified = eclf.predict(X_test_data)
    for i in range(0, len(test_tweets)):
   
        if classified[i] == test_tweets[i][1]:
            correct = correct + 1
            if classified[i] == "Positive":
                tp=tp+1
            elif classified[i] == "Negative":
                tn=tn+1

        if test_tweets[i][1] == "Positive":
            actual_positive=actual_positive+1
        elif test_tweets[i][1] == "Negative":
            actual_negative=actual_negative+1

        if classified[i] == "Positive":
            predicted_positive=predicted_positive+1
        elif classified[i] == "Negative":
            predicted_negative=predicted_negative+1

	
    # printing all numbers for test_data_classification   	
    print("Accuracy : ")
    print(correct/len(test_tweets)) 
    avg_accuracy=avg_accuracy+(correct/len(test_tweets))

    print("Positive Precision : ")
    pos_precision=pos_precision+(tp/predicted_positive)
    print(tp/predicted_positive)
    print("Negative Precision : ")
    neg_precision=neg_precision+(tn/predicted_negative)
    print((tn/predicted_negative))    

    print("Positive Recall : ")
    pos_recall=pos_recall+(tp/actual_positive)
    print(tp/actual_positive)
    print("Negative Recall : ")
    neg_recall=neg_recall+(tn/actual_negative)
    print((tn/actual_negative))  

    print("Positive class F-Score",( (2*pos_precision*pos_recall)/ ( (pos_precision)+(pos_recall) ) ))
    print("Negative class F-Score",( (2*neg_precision*neg_recall)/ ( (neg_precision)+(neg_recall) ) ))	
    	

	
	
	


print("Starting cleaning")		
#Some testing we did below in the commented code to check training data distribution
#positive=[]
#negative=[]
#neutral=[]
#for tweet,label in alltweets_list:
    #print(label)
    #if(label=="Positive"): positive.append((tweet,label))
    #elif(label=="Negative"): negative.append((tweet,label))
    #elif(label=="Neutral"): neutral.append((tweet,label)) 
#print("p",len(positive),"n",len(negative),"nl",len(neutral))  
#clean("test_data.csv")


#Cross validation for Obama
#clean("Obama.csv")
#shuffle(alltweets_list)
#cross_validation(alltweets_list,"Obama")

#Cross validation for Romney
#clean("Romney.csv")
#shuffle(alltweets_list)
#print("Total tweet List size for Romney:",len(alltweets_list))
#cross_validation(alltweets_list,"Romney")


##Testing data classification for Obama
clean("Obama.csv")
shuffle(alltweets_list)
train_tweets=alltweets_list
print("train_tweets_obama: ",len(train_tweets))
clean("testing-Obama.csv")
test_tweets=alltweets_list
print("test_tweets_obama: ",len(test_tweets))
test_data_classification(train_tweets,test_tweets,"Obama")

##Testing data classification for Romney  
clean("Romney.csv")
shuffle(alltweets_list)
train_tweets=alltweets_list
print("train_tweets_romney: ",len(train_tweets))
clean("testing-Romney.csv")
test_tweets=alltweets_list
print("test_tweets_romney: ",len(test_tweets))
test_data_classification(train_tweets,test_tweets,"Romney")
