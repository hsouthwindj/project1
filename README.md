# project1
the main function is the attack.py
you can run the basic experiment with following command

python -W ignore inprogressforkaggle.py --atk white --model rnn --l3  --iters 100 --group_size 9

below are the options for the program

`--atk, white box or black box setting, options : white, black`  
`--model, target video model type, options : rnn, c3d  `  
`--iters, maximum iteration number, default 100 for white box  `  
`--group_size, window size for white box attack  `  
`--overlap, windows overlap frame number  `  
`--l3, whether to use loss3 for white box, please always turn this on or transfer attack wont work  `  
`--mpath, model directory path  `  
`--dpath, data directory path  `  

## weights
you should be able to find the model weight in the cd
but here provides the source of the weight

you can download xeptionnet, meso from here [paarthneekhara github](https://github.com/paarthneekhara/AdversarialDeepFakes)  
for rnn+cnn model and related files, [PWB97 github](https://github.com/PWB97/Deepfake-detection)  
for 3dcnn model and related files, [NTech-Lab github](https://github.com/ntech-lab/deepfake-detection-challenge)  

for dataset, you can get both of them in kaggle  
(https://www.kaggle.com/competitions/deepfake-detection-challenge)  
(https://www.kaggle.com/datasets/sorokin/faceforensics)  

if there's any further question
you can email me aevin880713@gmail.com


