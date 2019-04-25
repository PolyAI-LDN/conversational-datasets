# Benchmark results on conversational response selection

This page contains benchmark results for the baselines and other methods on the datasets contained in this repository.

Please feel free to submit the results of your model as a pull request.

All the results are for models using only the `context` feature to select the correct `response`. Models using extra contexts are not reported here (yet).

For a description of the baseline systems, see [`baselines/README.md`](baselines/README.md).

## Reddit

These are results on the data from 2015 to 2018 inclusive,  (`TABLE_REGEX="^201[5678]_[01][0-9]$")$"`).


|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | 26.4%             	|
| BM25        	   | 27.5%             	|
| USE_SIM        	 | 36.6%             	|
| USE_MAP        	 | 40.8%             	|
| USE_LARGE_SIM    | 41.4%             	|
| USE_LARGE_MAP    | 47.7%             	|
| ELMO_SIM         | 12.5%             	|
| ELMO_MAP         | 20.6%             	|
| BERT_SMALL_SIM         | 14.4%             	|
| BERT_SMALL_MAP         | 20.5%             	|
| BERT_LARGE_SIM         | 10.6%             	|
| BERT_LARGE_MAP         | 17.1%             	|
| **Other models** |                    |
| PolyAI-Encoder [1]	  | 61.3%             	|


## OpenSubtitles

|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | 10.9%             	|
| BM25        	   | 10.9%             	|
| USE_SIM        	 | 13.6%             	|
| USE_MAP        	 | 15.8%             	|
| USE_LARGE_SIM    | 14.9%             	|
| USE_LARGE_MAP    | 18.0%             	|
| ELMO_SIM         | 9.5%             	|
| ELMO_MAP         | 13.3%             	|
| BERT_SMALL_SIM         | 13.8%             	|
| BERT_SMALL_MAP         | 17.3%             	|
| BERT_LARGE_SIM         | 9.9%             	|
| BERT_LARGE_MAP         | 14.2%             	|
| **Other models** |                    |
| PolyAI-Encoder [1]	  | 30.6%             	|


## AmazonQA

|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | 51.8%             	|
| BM25        	   | 52.3%             	|
| USE_SIM        	 | 47.6%             	|
| USE_MAP        	 | 54.4%             	|
| USE_LARGE_SIM    | 51.3%             	|
| USE_LARGE_MAP    | 61.9%             	|
| ELMO_SIM         | 16.0%             	|
| ELMO_MAP         | 35.5%             	|
| BERT_SMALL_SIM         | 16.1 %             	|
| BERT_SMALL_MAP         | 37.6 %             	|
| BERT_LARGE_SIM            | 12.3%          	|
| BERT_LARGE_MAP         | 30.2%             	|
| **Other models** |                    |
| PolyAI-Encoder [1]	  | 84.2%             	|


# References

[1] [A Repository of Conversational Datasets](https://arxiv.org/abs/1904.06472). Henderson et al. arXiv pre-print 2019.
