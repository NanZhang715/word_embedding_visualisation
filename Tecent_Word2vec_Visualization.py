#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:22:42 2018

@author: nzhang
"""
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import gensim
import codecs


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


fname='/home/zhangnan/data/Tencent_AILab_ChineseEmbedding.txt' 
model = gensim.models.KeyedVectors.load_word2vec_format(fname)

max_size = len(model.vocab)      
embed_dim = model.vector_size    


w2v = np.zeros((max_size,model.vector_size))
with codecs.open("metadata.tsv", 'w+','utf-8') as file_metadata:
        for i,word in enumerate(model.index2word[:max_size]):
            w2v[i] = model[word]
            file_metadata.write(word + '\n')
            
del w2v 

#Let us create a 2D tensor called embedding that holds our embeddings.
with tf.device("/cpu:0"):
    w = tf.Variable(tf.zeros(shape = [max_size, model.vector_size],dtype=tf.float32),
                    #trainable=True,
                    name ='w')
    
    embedding_placeholder = tf.placeholder(tf.float32, 
                                           shape=(max_size, model.vector_size),
                                           name = 'placholder')  
    
    embedding_init = w.assign(embedding_placeholder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(),feed_dict={embedding_init:model.vectors})
        
        path = 'tensorboard'
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(path,sess.graph)
        
        # add into projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embedding'
        embed.metadata_path = 'metadata.tsv'
        
        projector.visualize_embeddings(writer,config)
        
        saver.save(sess,path+'/model.ckpt')
    
    
            
