# dialog_paraphrase

# installations:
tensorflow == 1.15.0; tensorflow-probability == 0.7.0; python 3.6
Install pyrogue ( not rogue!!!)
Install nltk then stopwords, punkt, 
Install texar for transformer

quick start: 
	mkdir models
	mkdir outputs
	cd src
	python3 main.py --model_name=latent_bow --batch_size=5 --train_print_interval=10
	(change model_name=transformer_bow to try it out)