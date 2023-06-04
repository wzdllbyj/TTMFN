## Data

 Image features can be saved in npz format with clustering label, etc. 

The npz file contains the following information:

1. 'resnet-features': resnet features of each patch, 500*512
2. 'pid': the id of patient，A
3. 'time': OS time, 1000
4. 'status': OS status, 1
5. 'img_path': path of each patch, for example, './A/1.jpg'
6. 'cluster_num': cluster id of each patch. 500*1

Gene features of all patients can be seaved in csv file. The group of cluster is saved in txt file.

## Training

Our proposed TTMFN framework is in the `model.py ` file. After configuring the required environment, you can run:

```
python main.py
```

