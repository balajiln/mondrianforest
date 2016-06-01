This folder contains the scripts used in the following papers:

**Mondrian Forests: Efficient Online Random Forests**

Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh

*Advances in Neural Information Processing Systems (NIPS), 2014.*

[Link to PDF](http://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_nips14.pdf)

**Mondrian Forests for Large-Scale Regression when Uncertainty Matters**

Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh

*Proceedings of AISTATS, 2016.*

[Link to PDF](http://www.gatsby.ucl.ac.uk/~balaji/mfregression_aistats16.pdf)

Please cite the appropriate paper if you use this code.


I ran my experiments using Enthought python (which includes all the necessary python packages).
If you are running a different version of python (e.g. anaconda), you will need the following python packages 
(and possibly other packages) to run the scripts:

* numpy
* scipy
* matplotlib (for plotting Mondrian partitions)
* pydot and graphviz (for printing Mondrian trees)
* sklearn (for reading libsvm format files)

Some of the packages (e.g. pydot, matplotlib) are necessary only for '--draw_mondrian 1' option. If you just want to run experiments
without plotting the Mondrians, these packages may not be necessary.

Paul Heideman has created requirements.txt, which makes it easy to install the packages using 'pip install -r requirements.txt'.
Dan Stowell pointed out that dvipng package is required in ubuntu to draw the Mondrians.


The datasets are not included here; you need to download them from the UCI repository. You can run 
experiments using toy data though. Run **commands.sh** in **process_data** folder for automatically 
downloading and processing the datasets. I have tested these scripts only on Ubuntu, but it should be straightforward to process datasets in other platforms.

If you have any questions/comments/suggestions, please contact me at 
[balaji@gatsby.ucl.ac.uk](mailto:balaji@gatsby.ucl.ac.uk).

Code released under MIT license (see COPYING for more info).

Copyright &copy; 2014 Balaji Lakshminarayanan

----------------------------------------------------------------------------

**List of scripts in the src folder**:

- mondrianforest.py
- mondrianforest_utils.py
- mondrianforest_demo.py
- utils.py

I have added mondrianforest_demo.py which supports fit and partial_fit methods.

Help on usage can be obtained by typing the following commands on the terminal:

./mondrianforest.py -h

**Example usage**:

./mondrianforest_demo.py --dataset toy-mf --n_mondrians 100 --budget -1 --normalize_features 1 --optype class

**Examples that draw the Mondrian partition and Mondrian tree**:

./mondrianforest_demo.py --draw_mondrian 1 --save 1 --n_mondrians 10 --dataset toy-mf --store_every 1 --n_mini 6 --tag demo --optype class

./mondrianforest_demo.py --draw_mondrian 1 --save 1 --n_mondrians 1 --dataset toy-mf --store_every 1 --n_mini 6 --tag demo --optype class

**Example on a real-world dataset**:

*assuming you have successfully run commands.sh in process_data folder*

./mondrianforest_demo.py --dataset satimage --n_mondrians 100 --budget -1 --normalize_features 1 --save 1 --data_path ../process_data/ --n_minibatches 10 --store_every 1 --optype class

----------------------------------------------------------------------------

I generated commands for parameter sweeps using 'build_cmds' script by Jan Gasthaus, available publicly at [https://github.com/jgasthaus/Gitsby/tree/master/pbs/python](https://github.com/jgasthaus/Gitsby/tree/master/pbs/python).

Some examples of parameter sweeps are:

./build_cmds ./mondrianforest_demo.py "--op_dir={results}" "--init_id=1:1:6" "--dataset={letter,satimage,usps,dna,dna-61-120}" "--n_mondrians={100}" "--save={1}"  "--discount_factor={10.0}" "--budget={-1}" "--n_minibatches={100}" "--bagging={0}" "--store_every={1}" "--normalize_features={1}" "--data_path={../process_data/}" >> run

Note that the results (predictions, accuracy, log predictive probability on training/test data, runtimes) are stored in the pickle files. 
You need to write additional scripts to aggregate the results from these pickle files and generate the plots.
