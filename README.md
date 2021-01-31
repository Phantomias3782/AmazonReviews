# AmazonReviews
Classification of Amazon Reviews


## Authors

* **Andreas Bernrieder** - [Phantomias3782](https://github.com/Phantomias3782)
* **Thorsten Hilbradt** - [Thorsten-H](https://github.com/Thorsten-H)

## Getting Started

All code can be executed by the AmazonReviews.ipybn Jupyter Notebook. 

### use knn
When executing first time:
* Run all cells in the data preperation part and uncomment the save lines in the 
word vector part
* Run first knn cell, the n parameter is editable!
* For evaluation run 3. cell

When executing the second time you can load the data with saved word_vectors, just uncomment the corresponding line in the word vector part

### use neural network
When executing the first time:
* Run all cells in the data preperation part and uncomment the save lines in the 
word vector part
* Run first cell in the NN Part to create a model
* run 2nd and 3rd cells to predict user input.
When exectuing the scond time you can easily begin with the second cell in the NN part to load the model and idf dict.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

Special Thanks to:
* Justifying recommendations using distantly-labeled reviews and fined-grained aspects.

* Jianmo Ni, Jiacheng Li, Julian McAuley.

* Empirical Methods in Natural Language Processing (EMNLP), 2019.

* https://nijianmo.github.io/amazon/index.html

* http://deepyeti.ucsd.edu/jianmo/amazon/index.html
