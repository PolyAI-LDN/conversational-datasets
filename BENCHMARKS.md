
* all results for models using only the `context` feature to select the correct `response`. Models using extra contexts are not reported here (yet)
* (description of the baseline systems)[baselines/README.md]

## Reddit

Running on the data from January 2016 to June 2018 inclusive. This is the dataset used in **citation** .


|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | 22.5%             	|
| BM25        	   | 23.5%             	|
| USE_SIM        	 | 36.8%             	|
| USE_LARGE_SIM    | ??.?%             	|
| ELMO_SIM         | ??.?%             	|
| USE_MAP        	 | 41.1%             	|
| USE_LARGE_MAP    | ??.?%             	|
| ELMO_MAP         | ??.?%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 61.7%             	|


## OpenSubtitles

|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | 9.9%             	|
| BM25        	   | 9.9%             	|
| USE_SIM        	 | 13.6%             	|
| USE_LARGE_SIM    | ??.?%             	|
| ELMO_SIM         | ??.?%             	|
| USE_MAP        	 | 15.8%             	|
| USE_LARGE_MAP    | ??.?%             	|
| ELMO_MAP         | ??.?%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 19.1%             	|
| Encoder model **citation**  FT-DIRECT	  | 31.5%             	|

## AmazonQA

|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | 49.8%             	|
| BM25        	   | 50.2%             	|
| USE_SIM        	 | 47.5%             	|
| USE_LARGE_SIM    | 51.4%             	|
| ELMO_SIM         | ??.?%             	|
| USE_MAP        	 | 54.4%             	|
| USE_LARGE_MAP    | ??.?%             	|
| ELMO_MAP         | ??.?%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 61.6%             	|
| Encoder model **citation**  FT-DIRECT	  | 71.2%             	|
