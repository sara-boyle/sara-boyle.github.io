#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Sara Boyle

import sys
import numpy as np
import os
import shutil
import math
import random

def parse(text_file):
	#This function parses the text_file passed into it into a set of words. 
    #It just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)

dictionary_filename = "fold_dictionary.dict"
		
def makeMixedDirectory(spam_directory, ham_directory):
    #For the purposes of testing, this function randomly combines example spam and non-spam emails
    spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
    ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
    for f in ham:
        if (f == 'cmds'):
            ham.remove('cmds')
        if (f == 'desktop.ini'):
            ham.remove('desktop.ini')
    for f in spam:
        if (f == 'cmds'):
            spam.remove('cmds')
        if (f == 'desktop.ini'):
            spam.remove('desktop.ini')            

    combined_dictionary = ham[0:len(ham) - 1] + spam
    total_num_emails = len(combined_dictionary)    
    print total_num_emails    
    test_samples = random.sample(range(total_num_emails), total_num_emails)

    #Directories I'll have to make
    combined_directory = 'combined_directory/'
    new_spam_directory = 'new_spam_directory/'
    new_ham_directory = 'new_ham_directory/'
    ham_results = 'ham_results/'   
    spam_results = 'spam_results/'
    accuracy = []
    spam_count = 0
    ham_count = 0
    
    if not os.path.exists(combined_directory):   #This is a place to combine testing spam and ham material
        os.mkdir(combined_directory)        
    if not os.path.exists(new_spam_directory): #This is where to put the spam from the learning data
        os.mkdir(new_spam_directory)
    if not os.path.exists(new_ham_directory):  #This is where to put the ham from the learning data
        os.mkdir(new_ham_directory)
    if not os.path.exists(ham_results):  #This is where to put resulting ham from the testing data
        os.mkdir(ham_results)        
    if not os.path.exists(spam_results):  #This is where to put resulting spam from the testing data
        os.mkdir(spam_results)
        
# Split groups for running 6 fold testing        
    fold1 = test_samples[0:501]
    fold2 = test_samples[501:1001]
    fold3 = test_samples[1001:1501]
    fold4 = test_samples[1501:2001]
    fold5 = test_samples[2001:2501]
    fold6 = test_samples[2501:3001]
    folds = (fold1, fold2, fold3, fold4, fold5, fold6)
#==============================================================================
# For 30 fold testing:
#     fold1 = test_samples[0:101]
#     fold2 = test_samples[101:201]
#     fold3 = test_samples[201:301]
#     fold4 = test_samples[301:401]
#     fold5 = test_samples[401:501]
#     fold6 = test_samples[501:601]
#     fold7 = test_samples[601:701]
#     fold8 = test_samples[701:801]
#     fold9 = test_samples[801:901]
#     fold10 = test_samples[901:1001]
#     fold11 = test_samples[1001:1101]
#     fold12 = test_samples[1101:1201]
#     fold13 = test_samples[1201:1301]
#     fold14 = test_samples[1301:1401]
#     fold15 = test_samples[1401:1501]
#     fold16 = test_samples[1501:1601]
#     fold17 = test_samples[1601:1701]
#     fold18 = test_samples[1701:1801]
#     fold19 = test_samples[1801:1901]
#     fold20 = test_samples[1901:2001]
#     fold21 = test_samples[2001:2101]
#     fold22 = test_samples[2101:2201]
#     fold23 = test_samples[2201:2301]
#     fold24 = test_samples[2301:2401]
#     fold25 = test_samples[2401:2501]
#     fold26 = test_samples[2501:2601]
#     fold27 = test_samples[2601:2701]
#     fold28 = test_samples[2701:2801]
#     fold29 = test_samples[2801:2901]
#     fold30 = test_samples[2901:3001]
# 
#     folds = (fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10, fold11, fold12,
#              fold13, fold14, fold15, fold16, fold17, fold18, fold19, fold20, fold21, fold22, fold23, fold24,
#              fold25, fold26, fold27, fold28, fold29, fold30)  
#==============================================================================
    precisiontp = 0
    precisionfp = 0
    fn = 0
    for y in folds: #For each n fold,    
        if not os.path.exists(combined_directory):   #This is a place to combine testing spam and ham material
            os.mkdir(combined_directory)   
        if not os.path.exists(new_spam_directory): #This is where to put the spam from the learning data
            os.mkdir(new_spam_directory)
        if not os.path.exists(new_ham_directory):  #This is where to put the ham from the learning data
            os.mkdir(new_ham_directory)
        if not os.path.exists(ham_results):  #This is where to put resulting ham from the testing data
            os.mkdir(ham_results)        
        if not os.path.exists(spam_results):  #This is where to put resulting spam from the testing data
            os.mkdir(spam_results)  
        
        actual_answers = []
        for x in y: #Combine the spam and ham into a combined directory.
                    #This will constitute the test directory of both spam and ham.
            
            if (x > len(ham) - 2): #Then it is from spam
                shutil.copy(spam_directory + combined_dictionary[x], combined_directory) #Remember, you're making the combined
                actual_answers.append(1)                                                 #dictionary from the spam and the ham directory
        
            else:
                #ham_count += 1
                shutil.copy(ham_directory + combined_dictionary[x], combined_directory)
                actual_answers.append(0)
        for rest in folds:  #For all the other groups not including the one the test directory was from
                            #Copy all the items either to the spam directory or the ham directory
            if (rest != y):
                for j in rest:
                    if j > len(ham) - 2: #Copy spam to spam directory
                        shutil.copy(spam_directory + combined_dictionary[j], new_spam_directory)
                        spam_count += 1
                    else:
                        ham_count += 1
                        shutil.copy(ham_directory + combined_dictionary[j], new_ham_directory)        
        spam_count = 0
        ham_count = 0
        words, priorprob = makedictionary(new_spam_directory, new_ham_directory, dictionary_filename)
        tp = 0
        fp = 0
        ffn = 0
        #F1 = 2 * (precision * recall) / (precision + recall)
        spam_index = spamsort(combined_directory, spam_results, ham_results, words, priorprob)
        for a in range(len(actual_answers)):
            if (actual_answers[a] == 1):
                if (spam_index[a] == 1):
                    tp = tp + 1
                else:
                    ffn = ffn + 1
            else:
                if (spam_index[a] == 1):
                    fp = fp + 1
        
        for email in range(len(y)): #y is just the list of indeces, each fold
            if (y[email] > len(ham) - 2): #If the email is spam
                if(spam_index[email] == 1): #If your filter sorted the email as spam
                    precisiontp += 1   
                    #print precisiontp
                    accuracy.append(1)
                else: #If your filter wrongly labelled it as ham
                    fn += 1
                    accuracy.append(0)
            if (y[email] <= len(ham) - 2): #If the email is ham
                if(spam_index[email] == 0): #If your filter sorted the email as ham
                    accuracy.append(1)
                else: #If your filter wrongly labelled it as spam
                    precisionfp += 1
                    
                    accuracy.append(0)
                    
        shutil.rmtree('C:/Users/sarge/eecs349/hw5/new_spam_directory')
        shutil.rmtree('C:/Users/sarge/eecs349/hw5/new_ham_directory')
        shutil.rmtree('C:/Users/sarge/eecs349/hw5/spam_results')
        shutil.rmtree('C:/Users/sarge/eecs349/hw5/ham_results')
        shutil.rmtree('C:/Users/sarge/eecs349/hw5/combined_directory')
    print 'How many predictions? ' + str(len(accuracy))
    print 'tp: ' + str(tp)
    print 'fn: ' + str(ffn)
    print 'fp : ' + str(fp)
    precision = float(tp) / (tp + fp)
    print 'precision: ' + str(precision)
    recall = float(tp) / (tp + ffn)
    print 'recall: ' + str(recall)
    F1 = 2 * (precision * recall) / (precision + recall)
    print F1
    return accuracy, sum(accuracy)/float(len(accuracy)) #I will use this as a way to keep track of what in combined_directory is spam  
    
def makedictionary(spam_directory, ham_directory, dictionary_filename):
    """Making the dictionary. This returns a dictionary of email words that contains
    the probablity of the word being in a spam email. It also returns the prior
    probability of spam based on the training sets."""
    spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
    ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
    print 'Number of spam emails: ' + str(len(spam))
    print 'Number of ham emails: ' + str(len(ham))    

    total_num_emails = len(spam) + len(ham)
    print 'Total number of emails: ' + str(total_num_emails)    
    
    spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
    words = {}
    document_words = []
    #These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
    total_spam = 0 #Counting up all the spam documents so I can calculate p later
    total_ham = 0
    for s in spam:
        document_words = []
        total_spam = total_spam + 1
        for word in parse(open(spam_directory + s)):
            if word not in words:   #If the new word is one you've never seen before
                words[word] = {'spam': 1, 'ham': 0} #Set its spam value to 1. (I'm counting occurences right now)
                document_words.append(word)
            else:
                if (word not in document_words):
                    words[word] = {'spam': words[word]['spam'] + 1, 'ham': words[word]['ham']} #Adds 1 to spam value. (I'm counting occurences right now)
                    document_words.append(word)
                    
    for h in ham:
        document_words = []
        total_ham = total_ham + 1
        
        for word in parse(open(ham_directory + h)):
            if word not in words:
                words[word] = {'spam': 0, 'ham': 1}
                document_words.append(word)
            else:
                if (word not in document_words):                   #If it wasn't in the document before, add one to ham
                    words[word] = {'spam': words[word]['spam'], 'ham': words[word]['ham']  + 1}
                    document_words.append(word)
        
    for w in words:
        spam = words[w]['spam']
        ham = words[w]['ham']
        #print w
        words[w] = {'spam': float(spam)/total_spam, 'ham': float(ham)/total_ham}
	
    writedictionary(words, dictionary_filename)	
    return words, spam_prior_probability

def is_spam_lazy(content, words, spam_prior_probability):
    if spam_prior_probability >= .5:
        return True
    else:
        return False
def is_spam(content, words, spam_prior_probability):
    """This is the naive Bayes classifier."""
    spam_probabilities = []
    ham_probabilities = []
    document_words = []
    for word in content:
        if word in words and word not in document_words:
            spam_probabilities.append(words[word]['spam'])
            ham_probabilities.append(words[word]['ham'])
            document_words.append(word)        
    ss = 0        #I'm using equation 7
    for x in spam_probabilities:
        if (x != 0):
            ss = ss + math.log(x)
    spam_p = ss

    ss = 0
    for x in ham_probabilities:
        if (x != 0):
            ss = ss + math.log(x)
    ham_p = ss 
    
    ham_p = ham_p + math.log(1 - spam_prior_probability)
        
    spam_p = spam_p + math.log(spam_prior_probability)
    if (ham_p == 0) and (spam_p == 0):
            if spam_prior_probability >= .5:
                return True
            else:
                return False
    if (ham_p == 0):
        return True
    if (spam_p == 0):
        return False
    if (spam_p > ham_p):
        return True
    else:
        return False

    

def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
"""The function classifies each document in the mail directory as either spam or ham. Then, it copies the file
to spam_directory if it is spam. It copies the file to the ham_directory if it is ham (not spam)."""    
    mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]
    spam_count = 0
    ham_count = 0
    spam_index = [] #I'll use this to keep track of which emails were spam to test accuracy
    for m in mail:
        content = parse(open(mail_directory + m))
        spam = is_spam(content, dictionary, spam_prior_probability)
        #print spam
        if spam:
            #print mail_directory + m
            shutil.copy(mail_directory + m, spam_directory)
            spam_count += 1
            spam_index.append(1) #append 1 for spam
            #print 'Copied'
        else:
            ham_count += 1
            spam_index.append(0) #append 0 for ham
            shutil.copy(mail_directory + m, ham_directory)
    #print spam_index
    return spam_index

#==============================================================================
# sd = 'C:\Users\sarge\eecs349\hw5\sd\\'
# hd = 'C:\Users\sarge\eecs349\hw5\hd\\'
# words, p = makedictionary(spam_directory1, ham_directory1, 'dicty.txt')
# print spamsort(spam_directory1, sd, hd, words, p)
#==============================================================================

if __name__ == "__main__":
    #Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
    #training_spam_directory = sys.argv[1]
    #training_ham_directory = sys.argv[2]
    #test_mail_directory = sys.argv[3]

    training_spam_directory = 'C:/Users/sarge/eecs349/hw5/spam/'
    training_ham_directory = 'C:/Users/sarge/eecs349\hw5/easy_ham/'  
    test_mail_directory = 'C:/Users/sarge/eecs349\hw5/easy_ham/' 
    test_spam_directory = 'sorted_spam'
    test_ham_directory = 'sorted_ham'
	
    if not os.path.exists(test_spam_directory):
        os.mkdir(test_spam_directory)
    if not os.path.exists(test_ham_directory):
        os.mkdir(test_ham_directory)
#    spam_directory1 = 'C:/Users/sarge/eecs349/hw5/spam/'
#    ham_directory1 = 'C:/Users/sarge/eecs349/hw5/easy_ham/'
#    accuracy_v, accuracy = makeMixedDirectory(spam_directory1, ham_directory1)	
	dictionary_filename = "dictionary.dict"
	
	#create the dictionary to be used
	dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	#sort the mail
	spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability) 

