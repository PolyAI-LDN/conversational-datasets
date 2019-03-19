
* all results for models using only the `context` feature to select the correct `response`. Models using extra contexts are not reported here (yet)
* (description of the baseline systems)[baselines/README.md]

## Reddit

Running on the data from January 2016 to June 2018 inclusive. This is the dataset used in **citation** .


|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | ??.?%             	|
| BM25        	   | ??.?%             	|
| USE_SIM        	 | ??.?%             	|
| USE_LARGE_SIM    | ??.?%             	|
| ELMO_SIM         | ??.?%             	|
| USE_MAP        	 | ??.?%             	|
| USE_LARGE_MAP    | ??.?%             	|
| ELMO_MAP         | ??.?%             	|
| BM25        	   | ??.?%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 61.7%             	|


## OpenSubtitles

|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | ??.?%             	|
| BM25        	   | ??.?%             	|
| USE_SIM        	 | ??.?%             	|
| USE_LARGE_SIM    | ??.?%             	|
| ELMO_SIM         | ??.?%             	|
| USE_MAP        	 | ??.?%             	|
| USE_LARGE_MAP    | ??.?%             	|
| ELMO_MAP         | ??.?%             	|
| BM25        	   | ??.?%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 19.1%             	|
| Encoder model **citation**  FT-DIRECT	  | ??.?%             	|

## AmazonQA

|         	       | 1-of-100 accuracy 	|
| :---             | :---:	            |
| **Baselines**    |                    |         	
| TF_IDF           | ??.?%             	|
| BM25        	   | ??.?%             	|
| USE_SIM        	 | ??.?%             	|
| USE_LARGE_SIM    | 51.4%             	|
| ELMO_SIM         | ??.?%             	|
| USE_MAP        	 | ??.?%             	|
| USE_LARGE_MAP    | ??.?%             	|
| ELMO_MAP         | ??.?%             	|
| BM25        	   | ??.?%             	|
| **Other models** |                    |
| Encoder model **citation**  REDDIT-DIRECT	  | 61.6%             	|
| Encoder model **citation**  FT-DIRECT	  | 71.2%             	|
