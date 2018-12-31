# Embedding-Network
Embedding network for vehicle Re-identification.
This is a implementation of multi-loss (triplet loss + softmax loss) for vehicle ReID. 
For the triplet loss, we incorporate hard example mining in batch and soft-margin designed in paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737). 
Specially, you can also choose the normal triplet margin to optize the network. 
This project was modified based on the pedestrian re-identification project [triplet-reid-pytorch](https://github.com/CoinCheung/triplet-reid-pytorch).
The embedding network can be used to train on vehicle dataset, e.g., VehicleID and VeRI-776 dataset, which can get a higher performance than  [triplet-reid-pytorch](https://github.com/CoinCheung/triplet-reid-pytorch).

This project is based on pytorch0.4.0 and python3. 

Now the method of training on pretrained Resnet-50 is implemented. The VGGM model will be released soon. 


### prepare dataset
You need to download the VehicleID dataset, and put the dataset in "VehicleID/". 

### train the model
* To train on the VehicleID dataset, just run the training script:  
```
    $ cd scripts
    $ sh train_VehicleID.sh
```
This will train an embedder model based on ResNet-50. The trained model will be stored in the path of ```/res/model.pkl```.


### embed the test dataset
* To embed the test set, run the corresponding embedding scripts:
```
    $ cd scripts

    $ sh embed_trip_vehicleid.sh
```

### evaluate the embeddings for VehicleID dataset
* Then compute the rank-1 cmc and mAP:  
```
    $ cd scripts
    $ sh test_cmc.sh
    $ sh test_map.sh
```
This will evaluate the model with the query and gallery dataset. Note that, we use the VehicleID dataset CMC evaluation method, which gallery set only has one image for every vehicleIDs.

### evaluate the embeddings for VehicleID dataset 
   For VeRI-776 dataset:
   $ python veri_dist.py
   Then, you need to copy the dist file into veri_evaluate dir, and run the matlab ""


