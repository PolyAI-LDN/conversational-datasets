
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
| USE_MAP        	 | 41.1%             	|
| USE_LARGE_SIM    | 41.1%             	|
| USE_LARGE_MAP    | 47.9%             	|
| ELMO_SIM         | 11.6%             	|
| ELMO_MAP         | 16.9%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 61.7%             	|


## OpenSubtitles

|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | 9.9%             	|
| BM25        	   | 9.9%             	|
| USE_SIM        	 | 13.6%             	|
| USE_MAP        	 | 15.8%             	|
| USE_LARGE_SIM    | 14.9%             	|
| USE_LARGE_MAP    | 18.0%             	|
| ELMO_SIM         | 9.5%             	|
| ELMO_MAP         | 12.3%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 19.1%             	|
| Encoder model **citation**  FT-DIRECT	  | 31.5%             	|

## AmazonQA

|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | 49.8%             	|
| BM25        	   | 50.2%             	|
| USE_SIM        	 | 47.6%             	|
| USE_MAP        	 | 54.4%             	|
| USE_LARGE_SIM    | 51.3%             	|
| USE_LARGE_MAP    | 61.9%             	|
| ELMO_SIM         | 16.0%             	|
| ELMO_MAP         | 33.0%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 61.6%             	|
| Encoder model **citation**  FT-DIRECT	  | 71.2%             	|
