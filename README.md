# Anki Vector: Do Not Touch Your Face


To run the model on Vector, make sure you had the sdk set up.
then run this:

```
python vector_main.py
```


## Requirements: 
First make sure you can run the pose estimation model without any problem. If you can't, go to https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch and setup the enviroment.

Test to see if you can run pose estimation model:
```
python demo.py --checkpoint-path models/checkpoint_iter_370000.pth --images data/demo.jpg 
```
 
<p align="center">
  <img src="data/preview.jpg" />
</p>
_____________________________________________________
