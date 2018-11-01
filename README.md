# word_embedding_visualisation


Visualise Tencent_AILab_ChineseEmbedding With Tensorboard 



tensorboard --logdir tensorboard/


Notice 
- 1: tensorflow protocobuff 上限为2GB，故只使用 tf.placholder
- 2: ssh连接服务器，端口映射至本地，即可远程使用 tensorboard 
    具体方法如下：
    -  **连接ssh时，将服务器的6006端口重定向到自己机器上来：**
        - ssh -L 16006:127.0.0.1:6006 username@remote_server_ip
        - ssh -L 16006:127.0.0.1:6006 root@202.108.211.109 -p51050
        -  其中：16006:127.0.0.1代表自己机器上的16006号端口，6006是服务器上tensorboard使用的端口。

   - **在服务器上使用6006端口正常启动tensorboard：**
        - export LC_ALL=en_US.UTF-8
        - export LANGUAGE=en_US.UTF-8
        - tensorboard --logdir=xxx --port=6006
   
   - **在本地浏览器中输入地址**
       - 127.0.0.1:16006



