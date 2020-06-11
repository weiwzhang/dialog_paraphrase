# dialog_paraphrase

### installations:
- tensorflow == 1.15.0
- tensorflow-probability == 0.7.0. 
- python 3.6. 
- Install pyrogue ( not rogue!!!). 
- Install nltk==3.5 then download stopwords, punkt,   
- Install texar for transformer. 

download FastText at https://github.com/facebookresearch/fastText/, build repo, in repo run `./fasttext skipgram -input <input_txt> -output quora_fasttext_model -dim <dim>`


### quick start: 
	mkdir models
	mkdir outputs
	cd src
	python3 main.py --model_name=transformer_bow --batch_size=5 --train_print_interval=10
	(baseline: change model_name=latent_bow to try it out)
	

Note: this is a small experiment based on work by Fu et al., (2019) https://github.com/FranxYao/dgm_latent_bow/tree/master/src and Texar developers team (https://github.com/asyml/texar). 
