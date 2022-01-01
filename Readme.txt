%% Ant Colony Optimization (ACO) Image Feature Extraction Method
% Feature Extraction and Feature Selection are two different tasks. Feature
% Extraction is initial and vital step, but feature selection is optional.
% There are lots of evolutionary feature selection code are online for
% MATLAB but not feature extraction, especially for image. This code extracts
% features out of 10 classes of images with Ant Colony Optimization (ACO)
% evolutionary algorithm and compared it with extracted features using
% SURF with KNN classifier. Dataset is consists of 100 samples of small
% objects in 10 classes. You can use your data but labeling is done manually
% which you have to change it. following parameters are so important which
% you have to play with them in order to get desired results. Parameters
% are: 'nf', 'MaxIt', 'nAnt', knn classifier neighbors and number of hidden
% layers in "TrainNN.m" file. Feel free to contact me:
% Email: mosavi.a.i.buali@gmail.com
% Author: Seyed Muhammad Hossein Mousavi
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi
% This code is part of the following project, so if you used the code,
% please cite below paper:
% Mousavi, Seyed Muhammad Hossein, S. Younes MiriNezhad, and Mir Hossein Dezfoulian.
% "Galaxy gravity optimization (GGO) an algorithm for optimization, inspired by comets
% life cycle." 2017 Artificial Intelligence and Signal Processing Conference (AISP).
% IEEE, 2017.
% Hope it help you (Be Happy :)